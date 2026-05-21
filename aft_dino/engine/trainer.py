import torch
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule
from tqdm import tqdm


def ssl_train(model, dataloader, criterion, optimizer, EPOCHS):
    print("Starting Training")
    epochs = EPOCHS

    # Infer the active device from the model parameters.
    device = next(model.parameters()).device

    # Enable automatic mixed precision only on CUDA devices.
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Number of HAFD GateBlocks. It is zero when HAFD is disabled.
    num_gates = len(getattr(model, "student_gates", [])) if hasattr(model, "student_gates") else 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # DINO teacher-momentum cosine schedule.
        momentum_val = cosine_schedule(epoch, epochs, 0.996, 1.0)

        # Accumulate the mean gate value of each GateBlock during the epoch.
        gate_sums = torch.zeros(num_gates, device=device) if num_gates > 0 else None
        gate_counts = torch.zeros(num_gates, device=device) if num_gates > 0 else None

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for batch in pbar:
            views = batch[0]

            # Update the teacher backbone and projection head by EMA.
            update_momentum(model.student_backbone, model.teacher_backbone, m=momentum_val)
            update_momentum(model.student_head, model.teacher_head, m=momentum_val)

            # Update teacher-side HAFD adapters by EMA.
            if (
                hasattr(model, "student_adapters")
                and hasattr(model, "teacher_adapters")
                and len(model.student_adapters) == len(model.teacher_adapters)
                and len(model.student_adapters) > 0
            ):
                update_momentum(model.student_adapters, model.teacher_adapters, m=momentum_val)

            # Update teacher-side HAFD GateBlocks by EMA.
            if (
                hasattr(model, "student_gates")
                and hasattr(model, "teacher_gates")
                and len(model.student_gates) == len(model.teacher_gates)
                and len(model.student_gates) > 0
            ):
                update_momentum(model.student_gates, model.teacher_gates, m=momentum_val)

            views = [view.to(device) for view in views]
            global_views = views[:2]

            # The teacher receives global views; the student receives all views.
            with torch.cuda.amp.autocast(enabled=use_amp):
                teacher_out = [model.forward_teacher(view) for view in global_views]
                student_out = [model.forward(view) for view in views]

                # Compute TBA-DINO self-supervised loss in fp32 for stability.
                teacher_out = [t.float() for t in teacher_out]
                student_out = [s.float() for s in student_out]
                loss = criterion(teacher_out, student_out, epoch=epoch)

            total_loss += loss.detach().item()
            scaler.scale(loss).backward()

            # DINO stabilization: freeze the last projection layer early in training.
            model.student_head.cancel_last_layer_gradients(current_epoch=epoch)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Track the average gate values from the trainable student branch.
            if num_gates > 0 and hasattr(model, "last_student_gate_means") and model.last_student_gate_means:
                gate_vals = torch.tensor(model.last_student_gate_means, device=device)
                if gate_vals.numel() == num_gates:
                    gate_sums += gate_vals
                    gate_counts += 1

            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.5f}")

        if num_gates > 0 and gate_sums is not None and gate_counts is not None and torch.any(gate_counts > 0):
            avg_gates = (gate_sums / gate_counts.clamp(min=1)).tolist()
            print("Average student GateBlock values for this epoch:")
            for i, g in enumerate(avg_gates):
                print(f" GateBlock {i + 1}: {g:.4f}")

    return avg_loss, model
