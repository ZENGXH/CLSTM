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
conv = nn.SpatialConvolution(opt.nFiltersMemory[2], opt.nFiltersMemory[1], 3, 3, 1, 1, 1, 1)

dec:add(nn.Sequencer(decoder_0))
    :add(nn.Sequencer(decoder_1))
    :add(nn.Sequencer(conv))
--    :add(nn.Sequencer(decoderConv))


if opt.useGpu then
    require 'cunn'
    dec = dec:cuda()
    enc = enc:cuda()
    log.info('[model init] using gpu')
end
log.trace('encoder: ', enc)
log.trace('decoder:', dec)

-- encSeq = nn.Sequencer(enc)
-- decSeq = nn.Sequencer(dec)

if opt.backend == 'cudnn' then
  log.info('[init] using cudnn backend')
  require 'cudnn'
  enc = cudnn.convert(enc, cudnn)
  dec = cudnn.convert(dec, cudnn)
  cudnn.fastest = true 
else
  log.info('[init] NOT using cudnn backend')
end

model = nn.Sequential():add(enc):add(dec)
log:trace("[init] link model enc and dec done@")
