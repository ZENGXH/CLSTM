--[[
  Convolutional LSTM for short term visual cell
  inputSize - number of input feature planes
  outputSize - number of output feature planes
  rho - recurrent sequence length
  kc  - convolutional filter size to convolve input
  km  - convolutional filter size to convolve cell; usually km > kc  
--]]
local _ = require 'moses'
require 'nn'
require 'cunn'
local log = loadfile("log.lua")()
local backend_name = 'nn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end

local ConvLSTM, parent = torch.class('nn.ConvLSTM', 'nn.LSTM')

function ConvLSTM:__init(inputSize, outputSize, rho, kc, km, stride, batchSize)

   log.trace("[ConvLSTM] init input & outputSize ", inputSize, outputSize, 
            " kernel size ", kc, km, 
            "stride ", stride, 
            "padding ", padc, padm,
            "rho ", rho,
            "batchSize ", batchSize)
   self.kc = kc
   self.km = km
   self.padc = torch.floor(kc / 2)
   self.padm = torch.floor(km / 2)
   self.stride = stride or 1
   self.batchSize = batchSize or nil
   parent.__init(self, inputSize, outputSize, rho or 10)

   log.trace("[ConvLSTM] init input & outputSize ", self.inputSize, self.outputSize, 
            " kernel size ", self.kc, self.km, 
            "stride ", self.stride, 
            "padding ", self.padc, self.padm,
            "batchSize ", self.batchSize)
end

-------------------------- factory methods -----------------------------
function ConvLSTM:buildGateIF()
   -- Note : Input is : {input(t), output(t-1), cell(t-1)}
   log.trace("[ConvLSTM] start build GateIF")
   local gate = nn.Sequential()
   local input2gate = backend.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)
   local output2gate = backend.SpatialConvolution(self.outputSize, self.outputSize, self.km, self.km, self.stride, self.stride, self.padm, self.padm)
   WeightInit(input2gate)
   WeightInit(output2gate)
   input2gate = input2gate()
   output2gate = output2gate()

   output2gate:noBias()

   -- local cell2gate = nn.HadamardMul(self.inputSize, 1, 1) -- weight = inputSize x 1 x 1
   local cell2gate = backend.SpatialConvolution(self.outputSize, self.outputSize, 1, 1, self.stride, self.stride, 0, 0) -- weight = inputSize x 1 x 1
   local p, gp = cell2gate:getParameters()
   p = p:fill(0)
   p = p()
   local para = nn.ParallelTable()()
   para:add(input2gate):add(output2gate):add(cell2gate)
   gate:add(para)
   gate:add(backend.CAddTable())
   gate:add(backend.Sigmoid())
   log.trace("[ConvLSTM] @done build GateIF")
   return gate
end

function ConvLSTM:buildInputGate()
   -- i_t = sigmoid(W * x_t + W * h_t−1 + W o c_t−1 + bi)
   log.trace("[ConvLSTM] start build input gate")
   self.inputGate = self:buildGateIF()
   return self.inputGate
end

function ConvLSTM:buildForgetGate()
   -- f_t = sigmoid(W * x_t + W * h_t−1 + W o c_t−1 + bf)
   log.trace("[ConvLSTM] start build forget gate")
   self.forgetGate = self:buildGateIF()
   return self.forgetGate
end


function ConvLSTM:buildcellGate()
   -- Input is :{x_t, h_t-1, c-t-1}
   -- output is : Tanh(W * x-t + W * h_t-1 + b) = cellGate
   log.trace("[ConvLSTM] start buildcellGate")
   local hidden = nn.Sequential()
   hidden:add(nn.NarrowTable(1,2))
   local input2gate = backend.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)
   local output2gate = backend.SpatialConvolution(self.outputSize, self.outputSize, self.km, self.km, self.stride, self.stride, self.padm, self.padm)
   WeightInit(input2gate)
   WeightInit(output2gate)

   output2gate = output2gate:noBias()
   local para = nn.ParallelTable()
   para:add(input2gate):add(output2gate)
   hidden:add(para)
   hidden:add(nn.CAddTable())
   hidden:add(backend.Tanh())
   log.trace("[ConvLSTM] @done buildcellGate")

   return hidden
end

function ConvLSTM:buildcell()
   -- Input is : {input(t), output(t-1), cell(t-1)}
   -- Output is : forgetGate_t * c_t-1 + x_t * cellGate = cell_t
   log.trace("[ConvLSTM] start buildcell")
   self.inputGate = self:buildInputGate() 
   self.forgetGate = self:buildForgetGate()
   self.cellGate = self:buildcellGate()

   -- forget = forgetGate{input, output(t-1), cell(t-1)} * cell(t-1)
   local forget = nn.Sequential()

   local concat = nn.ConcatTable()
   concat:add(self.forgetGate):add(nn.SelectTable(3))

   forget:add(concat)
   forget:add(nn.CMulTable())

   -- input = inputGate{input(t), output(t-1), cell(t-1)} * cellGate{input(t), output(t-1), cell(t-1)}
   local input = nn.Sequential()
   
   local concat2 = nn.ConcatTable()
   concat2:add(self.inputGate):add(self.cellGate)
   input:add(concat2)
   input:add(nn.CMulTable())

   -- cell(t) = forget + input
   local cell = nn.Sequential()
   local concat3 = nn.ConcatTable()
   concat3:add(forget):add(input)
   cell:add(concat3)
   cell:add(nn.CAddTable())
   self.cell = cell
   log.trace("[ConvLSTM] @done buildcell")
   return cell
end   
   
function ConvLSTM:buildOutputGate()
   self.outputGate = self:buildGate()
   return self.outputGate
end

-- cell(t) = cell{input, output(t-1), cell(t-1)}
-- output of Model is table : {output(t), cell(t)} 
function ConvLSTM:buildModel()
    local inputs = {}
    table.insert(inputs, nn.Identity()())   -- network input
    table.insert(inputs, nn.Identity()())   -- c at time t-1
    table.insert(inputs, nn.Identity()())   -- h at time t-1
    local input = inputs[1]
    local prev_c = inputs[2]
    local prev_h = inputs[3]
    
    local i2h = nn.Linear(input_size, 4 * rnn_size)(input)  -- input to hidden
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)   -- hidden to hidden
    local preactivations = nn.CAddTable()({i2h, h2h})       -- i2h + h2h
    
    return model
end

function ConvLSTM:updateOutput(input)
   local prevOutput, prevCell
   
   if self.step == 1 then
      prevOutput = self.userPrevOutput or self.zeroTensor
      prevCell = self.userPrevCell or self.zeroTensor
      if self.batchSize then
         self.zeroTensor:resize(self.batchSize, self.outputSize, input:size(3), input:size(4)):zero()
      else
         self.zeroTensor:resize(self.outputSize,input:size(2),input:size(3)):zero()
      end
   else
      -- previous output and memory of this module
      prevOutput = self.output
      prevCell   = self.cell
   end
      
   -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
   local output, cell
   if self.train ~= false then
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      -- the actual forward propagation
      output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
   else
      output, cell = unpack(self.recurrentModule:updateOutput{input, prevOutput, prevCell})
   end
   
   self.outputs[self.step] = output
   self.cells[self.step] = cell
   
   self.output = output
   self.cell = cell
   
   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   self.gradParametersAccumulated = false
   -- note that we don't return the cell, just the output
   return self.output
end

function ConvLSTM:initBias(forgetBias, otherBias)
  local fBias = forgetBias or 1
  local oBias = otherBias or 0
  self.inputGate.modules[2].modules[1].bias:fill(oBias)
  self.outputGate.modules[2].modules[1].bias:fill(oBias)
  self.cellGate.modules[2].modules[1].bias:fill(oBias)
  self.forgetGate.modules[2].modules[1].bias:fill(fBias)
end
