import torch

CUDA = torch.cuda.is_available()


def epoch_train(model, num_epochs, epoch, data_loader, loss_fn, optimizer):
    model.train()
    sum_loss = 0
    sum_accuracy = 0
    iteration = 0
    train_epoch_loss = 0
    for i, (images, targets) in enumerate(data_loader):
        if CUDA:
            images = images.cuda()
            targets = targets.cuda()
        outputs, aux_out1, aux_out2 = model(images)
        _, predicted = torch.max(outputs, dim=1)
        _, aux1_predicted = torch.max(aux_out1, dim=1)
        _, aux2_predicted = torch.max(aux_out2, dim=1)
        if CUDA:
            sum_accuracy += (targets.cpu() == predicted.cpu()).sum()
        else:
            sum_accuracy += (targets == predicted).sum()
        loss = loss_fn(outputs, aux_out1, aux_out2, targets)
        sum_loss += loss.item() * len(targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iteration += 1
        train_epoch_loss = sum_loss / iteration
        if i % 10 == 0:
            print(
                "Epoch {}/{}, Iteration: {}/{}, Training loss: {:.3f}".format(
                    epoch + 1, num_epochs, i, len(data_loader), train_epoch_loss
                )
            )
    return train_epoch_loss


def epoch_test(model, num_epochs, epoch, data_loader, loss_fn):
    model.eval()
    sum_loss = 0
    sum_accuracy = 0
    iteration = 0
    train_epoch_loss = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if CUDA:
                images = images.cuda()
                targets = targets.cuda()
            outputs, aux_out1, aux_out2 = model(images)
            _, predicted = torch.max(outputs, dim=1)
            if CUDA:
                sum_accuracy += (targets.cpu() == predicted.cpu()).sum()
            else:
                sum_accuracy += (targets == predicted).sum()
            loss = loss_fn(outputs, None, None, targets)
            sum_loss += loss.item() * len(targets)
            iteration += 1
            test_epoch_accuracy = sum_accuracy / (iteration * len(images))
            test_epoch_loss = sum_loss / iteration
            print(
                "Epoch {}/{}, Iteration: {}/{}, Test loss: {:.3f}, Accuracy: {:.3f}".format(
                    epoch + 1,
                    num_epochs,
                    i,
                    len(data_loader),
                    test_epoch_loss,
                    test_epoch_accuracy,
                )
            )
    return test_epoch_loss, test_epoch_accuracy
