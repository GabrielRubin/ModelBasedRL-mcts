import torch
import torch.cuda as cuda
import pandas as pd
import numpy as np
import statistics
from pandas import DataFrame
from typing import List
from collections import OrderedDict
from data_manager import StateTransitionDataset
from tqdm import tqdm

gpu = torch.device("cuda")

class RNDModel(torch.nn.Module):

  def __init__(self, dimensions:List[int]):
    
    super().__init__()

    def Linear(i):
      return ('linear{0}'.format(i+1), torch.nn.Linear(dimensions[i], dimensions[i+1]))
    def ReLU(i):
      return ('ReLU{0}'.format(i+1), torch.nn.ReLU())

    modules1 = [func(i) for i in range(0, len(dimensions)-1) for func in (Linear, ReLU)]
    modules2 = [func(i) for i in range(0, len(dimensions)-1) for func in (Linear, ReLU)]
    del modules1[-1]
    del modules2[-1]

    self.predictionNetwork = torch.nn.Sequential(OrderedDict(modules1))
    self.targetNetwork     = torch.nn.Sequential(OrderedDict(modules2))
    self.ResetNetwork()

    #Freeze the target network parameters
    for param in self.targetNetwork.parameters():
      param.requires_grad = False

  def ResetNetwork(self):
    
    def WeightsInit(m):
      if isinstance(m, torch.nn.Linear):
        #torch.nn.init.kaiming_normal_(m, nonlinearity='relu') other option (?)
        torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
        m.bias.data.zero_()
    self.predictionNetwork.apply(WeightsInit)
    self.targetNetwork.apply(WeightsInit)

  def forward(self, state):
    predictionNetworkOutput = self.predictionNetwork(state)
    targetOutput            = self.targetNetwork(state)
    return predictionNetworkOutput, targetOutput

class NNComponents:

  def __init__(self, model, optimizer, lossFunction, loss=-1, epochCount=0, lastTestAcc=0):

    self.model        = model
    self.optimizer    = optimizer
    self.lossFunction = lossFunction
    self.loss         = loss
    self.epochCount   = epochCount
    self.lastTestAcc  = lastTestAcc

class NNBase:

  def __init__(self, dimensions:List[int], learningRate:float):

    self.dimensions   = dimensions
    self.learningRate = learningRate
    self.components   = None

  @classmethod
  def FromModel(cls, fileName:str, autoFormat:bool=True):
    
    if autoFormat:
      fileName = '{0}.pth'.format(fileName)
    checkpoint = torch.load(fileName)
    predictor  = cls(checkpoint['dimensions'],
                     checkpoint['learningRate'])

    predictor.components.model.load_state_dict(checkpoint['model_state_dict'])
    predictor.components.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    predictor.components.loss = checkpoint['loss']
    predictor.components.epochCount = checkpoint['epoch']
    predictor.components.lastTestAcc = checkpoint['lastTestAcc']

    return predictor

  def Save(self, fileName:str):

    if self.components is None:
      print('NNComponents not initialized!')
      return

    torch.save({'model_state_dict': self.components.model.state_dict(),
                'optimizer_state_dict': self.components.optimizer.state_dict(),
                'epoch': self.components.epochCount,
                'loss': self.components.loss,
                'dimensions': self.dimensions,
                'learningRate': self.learningRate,
                'lastTestAcc': self.components.lastTestAcc
                }, '{0}.pth'.format(fileName))

class StateNoveltyPredictor(NNBase):

  def __init__(self, dimensions:List[int], learningRate:float=1e-5):

    super().__init__(dimensions, learningRate)

    model = RNDModel(dimensions)
    model.to(gpu)
    optimizer    = torch.optim.Adam(model.parameters(), lr=self.learningRate)
    lossFunction = torch.nn.MSELoss(reduction='sum')

    self.components = NNComponents(model, optimizer, lossFunction)

  def Reset(self):
    self.components.model.ResetNetwork()

  def Train(self, trainData:DataFrame, epochCount:int=10):

    self.components.model.train()
    trainData  = trainData.astype(float)
    dimIn      = self.dimensions[ 0]
    dimOut     = self.dimensions[-1]

    x = torch.tensor(self.GetStateActionData(trainData).values.tolist(), device=gpu, dtype=torch.float)

    for t in range(epochCount):

      prediction, target = self.components.model(x)
      loss  = self.components.lossFunction(prediction, target)
      print(t, loss.item())

      self.components.optimizer.zero_grad()
      loss.backward()
      self.components.optimizer.step()
      self.components.loss = loss.item()

    self.components.epochCount += epochCount

  def GetNovelty(self, state:List[int]):

    self.components.model.train(False)
    self.components.model.eval()

    x = torch.tensor(state, device=gpu, dtype=torch.float)
    prediction, target = self.components.model(x)

    prediction = np.array(prediction.tolist())
    target     = np.array(target.tolist())

    mse = (np.square(prediction - target)).mean(axis=1)
    return sum(mse)

  # @TODO: REMOVE THIS METHOD, AND ALL ITS CALLS, ITS ONLY FOR TESTING, INPUT WILL BE JUST THE STATE
  def GetStateActionData(self, refData):
    return refData.drop(refData.columns[[i for i in 
                        range(self.dimensions[0], self.dimensions[0] + self.dimensions[1])]], axis=1)

  def GetStateActionData(self, refData):
    return refData.drop(refData.columns[[i for i in 
           range(self.stateActionLength, self.stateActionLength + self.succStateLength)]], axis=1)

  def Test(self, testData:DataFrame):
    testData      = testData.astype(float)
    inStates      = self.GetStateActionData(testData).values.tolist()
    return self.GetNovelty(inStates)

class StatePredictor(NNBase):

  def __init__(self, dimensions:List[int], learningRate:float=1e-4):
    
    super().__init__(dimensions, learningRate)

    self.stateActionLength = dimensions[0]
    self.succStateLength   = dimensions[-1]

    def Linear(i):
      return ('linear{0}'.format(i+1), torch.nn.Linear(self.dimensions[i], self.dimensions[i+1]))
    def ReLU(i):
      return ('ReLU{0}'.format(i+1), torch.nn.ReLU())

    modules = [func(i) for i in range(0, len(self.dimensions)-1) for func in (Linear, ReLU)]
    del modules[-1]
    
    model = torch.nn.Sequential(OrderedDict(modules))
    model.to(gpu)
    optimizer    = torch.optim.Adam(model.parameters(), lr=self.learningRate)
    lossFunction = torch.nn.MSELoss(reduction='sum')

    self.components = NNComponents(model, optimizer, lossFunction)

  def GetPrediction(self, state:List[int], action:List[int]):

    self.components.model.train(False)
    self.components.model.eval()

    x = torch.tensor(state + action, device=gpu, dtype=torch.float)
    result = self.components.model(x).tolist()

    return [float(j).__round__() for j in result]

  def GetPrediction2(self, stateAction:List[int]):

    self.components.model.train(False)
    self.components.model.eval()

    x = torch.tensor(stateAction, device=gpu, dtype=torch.float)
    result = self.components.model(x).tolist()

    return np.array([float(j).__round__() for j in result])

  def GetStateActionData(self, refData):
    return refData.drop(refData.columns[[i for i in 
           range(self.stateActionLength, self.stateActionLength + self.succStateLength)]], axis=1)

  def GetSuccStateData(self, refData):
    return refData.drop(refData.columns[[i for i in range(0, self.stateActionLength)]], axis=1)

  def Train(self, trainData:DataFrame, epochCount:int=5000):
    
    self.components.model.train()
    trainData  = trainData.astype(float)
    dimIn      = self.stateActionLength
    dimOut     = self.succStateLength

    x = torch.tensor(self.GetStateActionData(trainData).values.tolist(), device=gpu, dtype=torch.float)
    y = torch.tensor(self.GetSuccStateData(trainData).values.tolist(), device=gpu, dtype=torch.float)

    for t in range(epochCount):

      yPred = self.components.model(x)
      loss  = self.components.lossFunction(yPred, y)
      print(t, loss.item())

      self.components.optimizer.zero_grad()
      loss.backward()
      self.components.optimizer.step()
      self.components.loss = loss.item()

    self.components.epochCount += epochCount

  def Train2(self, trainDataset:StateTransitionDataset, batchSize:int=200, epochCount:int=10):

    self.components.model.train()
    
    datasetLoader = torch.utils.data.DataLoader(dataset=trainDataset,
                                                batch_size=batchSize,
                                                shuffle=True)
    pbar = tqdm(total=epochCount)
    last_loss  = 0
    loss_story = []
    for t in range(epochCount):
      for i, data in enumerate(datasetLoader):

        x = data[:, :self.stateActionLength].to(device=gpu, dtype=torch.float)
        y = data[:, self.stateActionLength:].to(device=gpu, dtype=torch.float)

        yPred = self.components.model(x)
        loss  = self.components.lossFunction(yPred, y)

        self.components.optimizer.zero_grad()
        loss.backward()
        self.components.optimizer.step()
        self.components.loss = loss.item()

      pbar.update(1)
      loss_value = loss.item()
      loss_story_value = 0
      if last_loss != 0:
        loss_story.append(loss_value - last_loss)
        loss_story_value = statistics.mean(loss_story)
      last_loss = loss_value
      if(len(loss_story) > 10):
        loss_story.pop(0)
      pbar.set_description("loss= {0:.1f} - delta= {1:.1f} ".format(loss_value, loss_story_value))

    self.components.epochCount += epochCount

  def Test(self, testData:DataFrame):

    testData      = testData.astype(float)
    inStateAction = self.GetStateActionData(testData).values.tolist()
    outState      = self.GetSuccStateData(testData).values.tolist()
    correct       = 0
    wrong         = 0

    x = torch.tensor(inStateAction, device=gpu, dtype=torch.float)
    y = torch.tensor(outState, device=gpu, dtype=torch.float)

    output = self.components.model(x).tolist()

    for i in range(len(output)):

        result = [float(j).__round__() for j in output[i]]
        actual = [float(j).__round__() for j in outState[i]]

        isEquals = True
        for j in range(len(actual)):
            if actual[j] != result[j]:
                isEquals = False
                break
        if isEquals:
            correct += 1
        else:
            wrong += 1

    self.components.lastTestAcc = (correct * 100.0) / (correct + wrong)
    print("======")
    print("ACCURACY = {0}%".format(self.components.lastTestAcc))
    print("CORRECT = {0}".format(correct))

  def Test2(self, trainDataset:StateTransitionDataset):

    correct = 0
    wrong   = 0
    
    datasetLoader = torch.utils.data.DataLoader(dataset=trainDataset,
                                                batch_size=50000,
                                                shuffle=False)

    dataSetSize = len(datasetLoader)
    
    for i, data in enumerate(datasetLoader):

      if i % (dataSetSize / 10) == 0:
        completition = (int)((i  / dataSetSize) * 10)
        print("TOTAL PROGRES: \r[{0}{1}] {2}%".format("=" * completition, " " * (10 - completition), (i  / dataSetSize) * 100))
      elif i == dataSetSize-1:
        print("TOTAL PROGRES: \r[{0}] 100%".format("=" * 10))

      print("dataset index = {0}".format(i))

      x = data[:, :self.stateActionLength].to(device=gpu, dtype=torch.float)

      print("x.")

      y = data[:, self.stateActionLength:].tolist()

      print("y..")

      output = self.components.model(x).tolist()

      print("output...")

      outputSize = len(output)

      for o in range(outputSize):

        if o % (outputSize / 10) == 0:
          completition = (int)((o  / outputSize) * 10)
          print("\r[{0}{1}] {2}%".format("=" * completition, " " * (10 - completition), (o  / outputSize) * 100))
        elif o == outputSize-1:
          print("\r[{0}] 100%".format("=" * 10))
          print("== PRE-RESULTS ==")
          print("ACCURACY = {0}%".format((correct * 100.0) / (correct + wrong)))
          print("CORRECT = {0}/{1}".format(correct, correct + wrong))

        result = [float(j).__round__() for j in output[o]]
        actual = [float(j).__round__() for j in y[o]]

        isEquals = True
        for j in range(len(actual)):
            if actual[j] != result[j]:
                isEquals = False
                break
        if isEquals:
            correct += 1
        else:
            wrong += 1

    self.components.lastTestAcc = (correct * 100.0) / (correct + wrong)
    print("== FINAL RESULTS ==")
    print("ACCURACY = {0}%".format(self.components.lastTestAcc))
    print("CORRECT = {0}/{1}".format(correct, correct + wrong))

    return self.components.lastTestAcc
