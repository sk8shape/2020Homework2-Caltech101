since = time.time()
net = net.to(DEVICE)
cudnn.benchmark
current_step = 0
for epoch in range(NUM_EPOCHS):
  #print('Starting epoch {}/{}, LR = {}'.format(epoch+1, NUM_EPOCHS, scheduler.get_lr()))
  for phase in ['train', 'val']:
    if phase == 'train':
      net.train(True)  # Set model to training mode
    else:
      net.train(False)   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    for images, labels in train_dataloader:
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      optimizer.zero_grad()
      #forward
      outputs = net(images)

      with torch.set_grad_enabled(phase == 'train'):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        if phase == 'train':
          loss.backward()
          optimizer.step()

      if phase == 'train':
          scheduler.step()
          current_step += 1

      # Log loss
      if current_step % LOG_FREQUENCY == 0:
        print('Step {}, Loss {}'.format(current_step, loss.item()))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
return model
      
