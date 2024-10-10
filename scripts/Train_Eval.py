from imports import*
from utils import*
from models import*

class Train():

  def __init__(self, model, device, data_loader, optimizer, loss_f):
        
    self.model = model
    self.device = device
    self.data_loader = data_loader
    self.optimizer = optimizer
    self.loss_f = loss_f

  def train_function(self):

    # Sets model to train mode
    self.model.train()

    loss_ = 0

    count = 1

    model = self.model.to(device)

    for step, batch in enumerate(tqdm(self.data_loader, desc="Iteration")): #remind that tqdm draws progress bars


      batch = batch.to(device)
      edge_index = (batch.edge_index).type(torch.LongTensor).to(device) # for simple GNN
      adj_t = batch.adj_t.to(device) # for ViG

      out = model((batch.x).float(), edge_index) #Feed the data into the model
      loss_ = self.loss_f(out, batch.y.to(torch.int64))

      # backpropagate

      if count % 16 == 0: # we backpropagate every 16 steps to simulate a batch size of 64
 
        loss_.backward()
        self.optimizer.step()
        self.optimizer.zero_grad() #Zero grad the optimizer

      else:

        loss_.backward()

      count += 1  
      
      torch.cuda.empty_cache()
      del(batch)
      del(adj_t)
      del(edge_index)
      gc.collect()

      if step != stop-2: # we eliminate the loss vector to free memory for each step apart from the last one
        del(loss_)
        gc.collect()

    return loss_.item()
  
class Eval:

  def __init__(self, model, device, loss_f):
        
    self.model = model
    self.device = device
    self.loss_f = loss_f

  def eval_function(self, data_loader, Dice):

    # Sets model to eval mode
    self.model.eval()
    data_loader = data_loader

    dice_per_class = []

    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")): #remind that tqdm draws progress bars

      model = self.model.to(device)

      batch = batch.to(device)

      edge_index = batch.edge_index.type(torch.LongTensor).to(device) # for simple GNN

      adj_t = batch.adj_t.to(device) # for ViG

      with torch.no_grad():
        out = model((batch.x).float(), edge_index) #Feed the data into the model
        loss_ = self.loss_f(out, batch.y.to(torch.int64))

        if Dice == True:
        
          pred = torch.argmax(model((batch.x).float(), edge_index), axis=1)
          y_true = batch.y.view(pred.shape)
          y_pred = pred

          dice_per_class.append(Tensor.numpy(dice(y_pred, y_true, average='none', num_classes=args['num_classes']).detach().cpu()))
            
      
      torch.cuda.empty_cache()
      del(batch)
      del(edge_index)
      del(adj_t)
      gc.collect()

    mean_dice_per_class = np.mean(np.array(dice_per_class), axis=0)

    return loss_.item(), mean_dice_per_class