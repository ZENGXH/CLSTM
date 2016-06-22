-- use nn.graph to build ConvLSTM.lua
-- inspired by FastLSTM.lua in rnn

-- set this to true to have it use nngraph instead of nn
-- setting this to true can make your next FastLSTM significantly faster

require 'nngraph'
require 'nn'
require 'ConvLSTM'
require 'dpnn'
require 'rnn'
local FastConvLSTM, parent = torch.class("nn.FastConvLSTM", "nn.ConvLSTM")

FastConvLSTM.usenngraph = true -- dafault to be true

function FastConvLSTM:__init(inputSize, outputSize, rho, kc, km, stride, batchSize, hight, width)
  self.H = hight or 50
  self.W = width or 50
   -- (inputSize, outputSize, rho, kc, km, stride, batchSize)
  self.kc = kc or 3
  self.km = km or 3
  self.stride = stride or 1
  self.inputSize = inputSize
  self.outputSize = outputSize
  self.batchSize = batchSize
  self.i2g_padding = math.floor(kc / 2) or 1
  self.o2g_padding = math.floor(km / 2) or 1
  self.rho = rho or 9999
  parent.__init(self, self.inputSize, self.outputSize, self.rho, self.kc, self.km, self.stride, self.batchSize)

end

function FastConvLSTM:buildModel()
      self.i2g = nn.SpatialConvolution(self.inputSize, self.outputSize*4, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)
      self.o2g = nn.SpatialConvolution(self.outputSize, self.outputSize*4, self.km, self.km, self.stride, self.stride, self.padm, self.padm) 
    if self.usenngraph then
		  return self:nngraphModel()
	  else
		  -- return nn.ConvLSTM(self.inputSize, self.outputSize, self.rho, self.kc, self.km, self.stride, self.batchSize)
---------------------------------------
    print('not using nn graph')
   local para = nn.ParallelTable():add(self.i2g):add(self.o2g)

   gates = nn.Sequential()
   gates:add(nn.NarrowTable(1,2))
   gates:add(para)
   gates:add(nn.CAddTable())

   -- Reshape to (batch_size, n_gates, hid_size)
   -- Then slize the n_gates dimension, i.e dimension 2
   --print('reshape:: ')
   --print(self.batchSize, 4, self.outputSize, self.H, self.W)
   gates:add(nn.Reshape(self.batchSize, 4, self.outputSize, self.H, self.W))
   gates:add(nn.SplitTable(2))
   transfer = nn.ParallelTable()
   transfer:add(nn.Sigmoid()):add(nn.Tanh()):add(nn.Sigmoid()):add(nn.Sigmoid())
   gates:add(transfer)

   local concat = nn.ConcatTable()
   concat:add(gates):add(nn.SelectTable(3))
   local seq = nn.Sequential()
   seq:add(concat)
   seq:add(nn.FlattenTable()) -- input, hidden, forget, output, cell
   
   -- input gate * hidden state
   local hidden = nn.Sequential()
   hidden:add(nn.NarrowTable(1,2))
   hidden:add(nn.CMulTable())
   
   -- forget gate * cell
   local cell = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(nn.SelectTable(3)):add(nn.SelectTable(5))
   cell:add(concat)
   cell:add(nn.CMulTable())
   
   local nextCell = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(hidden):add(cell)
   nextCell:add(concat)
   nextCell:add(nn.CAddTable())
   
   local concat = nn.ConcatTable()
   concat:add(nextCell):add(nn.SelectTable(4))
   seq:add(concat)
   seq:add(nn.FlattenTable()) -- nextCell, outputGate
   
   local cellAct = nn.Sequential()
   cellAct:add(nn.SelectTable(1))
   cellAct:add(nn.Tanh())
   local concat = nn.ConcatTable()
   concat:add(cellAct):add(nn.SelectTable(2))
   local output = nn.Sequential()
   output:add(concat)
   output:add(nn.CMulTable())
   
   local concat = nn.ConcatTable()
   concat:add(output):add(nn.SelectTable(1))
   seq:add(concat)
   
   return seq


----------------------------------------
    end
end
-- rho, kc, km, stride, batchSize

function FastConvLSTM:nngraphModel()
   print('using nn graph')
   assert(nngraph, "Missing nngraph package")
   
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x
   table.insert(inputs, nn.Identity()()) -- prev_h[L]
   table.insert(inputs, nn.Identity()()) -- prev_c[L]
   
   local x, prev_h, prev_c = unpack(inputs)
   
   -- evaluate the input sums at once for efficiency
   local i2h = self.i2g(x):annotate{name='i2h'}
   local h2h = self.o2g(prev_h):annotate{name='h2h'}
   local all_input_sums = nn.CAddTable()({i2h, h2h})

   local reshaped = nn.Reshape(4, self.outputSize, self.H, self.W)(all_input_sums)
   -- input, hidden, forget, output
   local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
   local in_gate = nn.Sigmoid()(n1)
   local in_transform = nn.Tanh()(n2)
   local forget_gate = nn.Sigmoid()(n3)
   local out_gate = nn.Sigmoid()(n4)
   
   -- perform the LSTM update
   local next_c           = nn.CAddTable()({
     nn.CMulTable()({forget_gate, prev_c}),
     nn.CMulTable()({in_gate,     in_transform})
   })
   -- gated cells form the output
   local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

   local outputs = {next_h, next_c}
   mlp = nn.gModule(inputs, outputs)

   return mlp
end