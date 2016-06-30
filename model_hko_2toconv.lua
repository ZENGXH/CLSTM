require 'rnn'
require 'ConvLSTM'
require 'utils'
local log = loadfile("log.lua")()

-- encoder_0 = nn.ConvLSTM(opt.inputSeqLen, opt.inputSeqLen, opt.encoderRho, opt.nFiltersMemory[1], opt.nFiltersMemory[2], opt.stride, opt.batchSize)
-- encoder_1 = nn.ConvLSTM(opt.inputSeqLen, opt.inputSeqLen, opt.encoderRho, opt.nFiltersMemory[2], opt.nFiltersMemory[2], opt.stride, opt.batchSize)
-- decoder_0 = nn.ConvLSTM(opt.outputSeqLen, opt.outputSeqLen, opt.decoderRho, opt.nFiltersMemory[1], opt.nFiltersMemory[2], opt.stride, opt.batchSize)
-- decoder_1 = nn.ConvLSTM(opt.outputSeqLen, opt.outputSeqLen, opt.decoderRho, opt.nFiltersMemory[2], opt.nFiltersMemory[2], opt.stride, opt.batchSize)

encoder_0 = nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2], opt.encoderRho, 3, 3, opt.stride, opt.batchSize)
encoder_1 = nn.ConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2], opt.encoderRho, 3, 3, opt.stride, opt.batchSize)
decoder_0 = nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2], opt.decoderRho, 3, 3, opt.stride, opt.batchSize)
decoder_1 = nn.ConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2], opt.decoderRho, 3, 3, opt.stride, opt.batchSize)

enc = nn.Sequential()
enc.lstmLayers = {}
enc.lstmLayers[1] = encoder_0
enc.lstmLayers[2] = encoder_1

dec = nn.Sequential()
dec.lstmLayers = {}
dec.lstmLayers[1] = decoder_0
dec.lstmLayers[2] = decoder_1

-- enc:add(encoder_0):add(encoder_1)
enc:add(nn.Sequencer(encoder_0)):add(nn.Sequencer(encoder_1))
-- dec:add(decoder_0):add(decoder_1)
conv = nn.SpatialConvolution(opt.nFiltersMemory[2] * 2, opt.nFiltersMemory[1], 3, 3, 1, 1, 1, 1)

-- decoderConv = nn.Sequential():add(decoder_1):add(conv)


decSeq = nn.Sequential():add(decoder_0)
local concat = nn.ConcatTable()
concat:add(decoder_1):add(nn.Identity())
decSeq:add(concat):add(nn.JoinTable(2))

decSeq:add(conv)

dec:add(nn.Sequencer(decSeq))


if opt.useGpu then
    require 'cunn'
    dec = dec:cuda()
    enc = enc:cuda()
    log.info('[model init] using gpu')
end
local backend_name = opt.backend or 'cudnn'

local backend
if backend_name == 'cudnn' then
  log.info('[init] using cudnn backend')
  require 'cudnn'
  backend = cudnn
  encoder_0 = cudnn.convert(encoder_0, cudnn)
  encoder_1 = cudnn.convert(encoder_1, cudnn)
  decoder_0 = cudnn.convert(decoder_0, cudnn)
  decoder_1 = cudnn.convert(decoder_1, cudnn)
  conv = cudnn.convert(conv, cudnn)
  cudnn.fastest = true 
else
  backend = nn
  log.info('[init] NOT using cudnn backend')
end

log.trace('encoder: ', enc)
log.trace('decoder:', dec)

-- encSeq = nn.Sequencer(enc)
-- decSeq = nn.Sequencer(dec)

log:trace("[init] load model enc and dec done@")
model = nn.Sequential()
model:add(enc):add(dec)
