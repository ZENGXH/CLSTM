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
require 'utils'
local log = loadfile("log.lua")()
local backend = nn
local ConvLSTM, parent = torch.class('nn.ConvLSTM', 'nn.Container')

function ConvLSTM:__init(inputSize, outputSize, rho, kc, km, stride, batchSize)
    self.kc = kc
    self.km = km
    self.padc = torch.floor(kc / 2)
    self.padm = torch.floor(km / 2)
    self.stride = stride or 1
    self.batchSize = batchSize or nil
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.zeroTensor = torch.Tensor()
    log.trace("[ConvLSTM] init input & outputSize ", self.inputSize, self.outputSize, 
    " kernel size ", self.kc, self.km, 
    "stride ", self.stride, 
    "padding ", self.padc, self.padm,
    "batchSize ", self.batchSize)
    self.modules = {}
    self.module = self:buildModel()
    table.insert(self.modules, self.module)
end

-------------------------- factory methods -----------------------------
function ConvLSTM:buildGateIF()
   -- Note : Input is : {input(t), output(t-1), cell(t-1)}
   log.trace("[ConvLSTM] start build GateIF")
   gate = nn.Sequential()
   local input2gate = backend.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)
   local output2gate = backend.SpatialConvolution(self.outputSize, self.outputSize, self.km, self.km, self.stride, self.stride, self.padm, self.padm)
   WeightInit(input2gate)
   WeightInit(output2gate)

   output2gate:noBias()

   -- local cell2gate = nn.HadamardMul(self.inputSize, 1, 1) -- weight = inputSize x 1 x 1
   local cell2gate = backend.SpatialConvolution(self.outputSize, self.outputSize, 1, 1, self.stride, self.stride, 0, 0) -- weight = inputSize x 1 x 1
   local p, gp = cell2gate:getParameters()
   p = p:fill(0)
   local para = nn.ParallelTable()
   para:add(input2gate):add(output2gate):add(cell2gate)
   gate:add(para)
   gate:add(backend.CAddTable())
   gate:add(backend.Sigmoid())
   log.trace("[ConvLSTM] @done build GateIF")
   return gate
end

function ConvLSTM:buildGate()
   -- Note : Input is : {input(t), output(t-1), cell(t-1)}
   log.trace("[ConvLSTM] start build GateIF")
   local gate = nn.Sequential()
   local input2gate = backend.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)
   local output2gate = backend.SpatialConvolution(self.outputSize, self.outputSize, self.km, self.km, self.stride, self.stride, self.padm, self.padm)
   WeightInit(input2gate)
   WeightInit(output2gate)

   output2gate:noBias()

   -- local cell2gate = nn.HadamardMul(self.inputSize, 1, 1) -- weight = inputSize x 1 x 1
   -- local cell2gate = backend.SpatialConvolution(self.outputSize, self.outputSize, 1, 1, self.stride, self.stride, 0, 0) -- weight = inputSize x 1 x 1
   -- local p, gp = cell2gate:getParameters()
   p = p:fill(0) -- peephole's weight always set to zero at the begining
   local para = nn.ParallelTable()
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
   -- Input is : {input(t), output(t-1), cell(t-1)}
   log.trace("[ConvLSTM] start buildModel")

   local cell = self:buildcell()
    
   local concat = nn.ConcatTable()
   concat:add(nn.NarrowTable(1,2)):add(cell)   

   -- < add to model >
   local model = nn.Sequential()
   model:add(concat)
   model:add(nn.FlattenTable())
   -- output of concat is {{input(t), output(t-1)}, cell(t)}, 
   -- so flatten to {input(t), output(t-1), cell(t)}

   -- cellAct = Tanh( cell_(t)<current> )
   local cellAct = nn.Sequential()
   cellAct:add(nn.SelectTable(3))
   cellAct:add(backend.Tanh())

   -- self.outputGate = W * input + W * output(t - 1) + W * cell_(t)<current> + b
   self.outputGate = self:buildGateIF() 
   
   -- output(t) = W * {input, output(t-1), cell(t)} * tanh(cell(t))
   -- ie output(t) = outputGate x cellAct
   local concat3 = nn.ConcatTable()
   concat3:add(self.outputGate):add(cellAct)
   local output = nn.Sequential()
   output:add(concat3)
   output:add(nn.CMulTable())

   -- we want the model to output : {output(t), cell(t)}
   local concat4 = nn.ConcatTable()
   concat4:add(output):add(nn.SelectTable(3))
 
   -- < add to model>
   model:add(concat4)
   return model
end

function ConvLSTM:updateOutput(input)
   log.trace('[ConvLSTM] updateOutput ')
   if(not self.prevOutput) then
       log.trace('[ConvLSTM] no memory')
       self.preCell = prevCell
       self.prevOutput = prevOutput
   end

   self.prevOutput = self.prevOutput or self.zeroTensor
   self.prevCell = self.prevCell or self.zeroTensor



   if self.batchSize then
       self.zeroTensor:resize(self.batchSize, self.outputSize, input:size(3), input:size(4)):zero()
   else
       self.zeroTensor:resize(self.outputSize,input:size(2),input:size(3)):zero()
   end

   -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}

   self.output, self.cell = unpack(self.module:updateOutput({input, self.prevOutput, self.prevCell}))
   assert(self.output)
   assert(self.cell)
   -- self.prevCell = nil
   -- note that we don't return the cell, just the output
   return self.output
end
function ConvLSTM:updateGradInput(input, gradOutput)

    self.gradCell = self.gradPrevCell or torch.Tensor():resizeAs(self.cell)

    log.trace(string.format('[ConvLSTM] updateGradInput get input %f prevOutput %f, prevCell %f', input:mean(), self.prevOutput:mean(), self.prevCell:mean()))         
    self.gradInput, gradPrevOutput, gradPrevCell = unpack(
        self.module:updateGradInput(
            {input, self.prevOutput, self.prevCell}, 
            {gradOutput, self.gradCell})
            )

    return self.gradInput
end

function ConvLSTM:accGradParameters(input, gradOutput)
       log.trace(string.format('[ConvLSTM] accGradParameters: input table:%f, %f, %f ', input:mean(), self.prevOutput:mean(), self.prevCell:mean()))

       log.trace(string.format('[ConvLSTM] accGradParameters: grad table: %f, %f ', gradOutput:mean(), self.gradCell:mean()))

       self.module:accGradParameters(
            {input, self.prevOutput, self.prevCell}, 
            {gradOutput, self.gradCell})

       local para, grad = self.module:getParameters()
       log.trace(string.format('[ConvLSTM] update loss of cell %f and output %f gradient %f ', self.gradCell:mean(), gradOutput:mean(), grad:mean()))

end


function ConvLSTM:initBias(forgetBias, otherBias)
  local fBias = forgetBias or 1
  local oBias = otherBias or 0
  self.inputGate.modules[1].modules[1].bias:fill(oBias)
  self.outputGate.modules[1].modules[1].bias:fill(oBias)
  self.cellGate.modules[2].modules[1].bias:fill(oBias)
  self.forgetGate.modules[1].modules[1].bias:fill(fBias)
end

function ConvLSTM:clearMemory()
    if(torch.isTensor(self.cell)) then
        self.cell = nil 
    end
end


