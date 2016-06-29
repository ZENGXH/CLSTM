
require 'ConvLSTMAlong'
require 'opts_hko.lua'

net = nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2], opt.rho, 3, 3, 1, opt.batchSize)
inputTable = {}
log = loadfile('log.lua')()
log.trace('[init] set up ConvLSTM, ', net)

local input

for i = 1, 5 do
    input = torch.randn(1, 10000):resize(opt.batchSize, 1, 100, 100)
    table.insert(inputTable, input)
end

net:zeroGradParameters()
local outputTable = {}
local para_, grad_ = net:getParameters()
local para = para_:clone()
local grad = grad_:clone()
log.trace('size of grad: ', para:size(), grad:size())
assert(grad:mean() == 0)
local output 
log.trace(string.format('[start] para: %f grad: %f', para:mean(), grad:mean()))
for i = 1, #inputTable do
  input = inputTable[i]
  output = net:updateOutput(input)
  table.insert(outputTable, output)
  local gradOutput = torch.randn(1, 10000):resizeAs(output)
  assert(gradOutput:mean() ~= 0)

  net.prevCell:fill(0.2)
  net.prevOutput:fill(0.2)
  
  net:updateGradInput(input, gradOutput)
  assert(input:mean() ~= 0)
  net:accGradParameters(input, gradOutput)

  net.prevCell = net.cell
  net.prevOutput = output
  log.trace('iter: ', i, ' output: ', output:mean())
  local para2, grad2 = net:getParameters()
  local para_module, grad_module = net.module:getParameters()
  assert(grad_module:mean() == grad2:mean())

  log.trace(string.format('grad: of new net %f, old net %f, net module %f ', grad2:mean(), grad:mean(), grad_module:mean()))
  assert(grad2:mean() ~= grad:mean(), 'grad after update still zero')
  assert(para2:mean() == para2:mean())
  grad = grad2:clone()
  para = para2:clone()
end

log.trace(output:size())
net:updateParameters(opt.lr)
local para2, grad2 = net:getParameters()
assert(grad2:mean() ~= grad:mean())
assert(para2:mean() ~= para2:mean())

net:zeroGradParameters()
para, grad = net:getParameters()
assert(grad:mean() == 0)
