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

decoderConv = nn.Sequential():add(decoder_1):add(conv)

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
log.trace(enc)
log.trace(dec)

-- encSeq = nn.Sequencer(enc)
-- decSeq = nn.Sequencer(dec)

log:trace("[init] load model enc and dec done@")
function forwardConnect(enc, dec)
    -- log.trace('[forwardConnect]')
    for i=1, #enc.lstmLayers do
        if opt.useSeqLSTM then
            dec.lstmLayers[i].userPrevOutput = enc.lstmLayers[i].output[opt.inputSeqLen]
            dec.lstmLayers[i].userPrevCell = enc.lstmLayers[i].cell[opt.inputSeqLen]
        else
            -- log.info("dec connect: ", #enc.lstmLayers[i].cells)
            assert(#enc.lstmLayers[i].cells == opt.inputSeqLen)
            dec.lstmLayers[i].userPrevOutput = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevOutput, enc.lstmLayers[i].outputs[opt.inputSeqLen])
            dec.lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevCell, enc.lstmLayers[i].cells[opt.inputSeqLen])
        end
    end
    --log.trace('[forwardConnect] done@')
end

function backwardConnect(enc, dec)
    -- log.trace('[backwardConnect]')
    for i=1,#enc.lstmLayers do
        if opt.useSeqLSTM then
            enc.lstmLayers[i].userNextGradCell = dec.lstmLayers[i].userGradPrevCell
            enc.lstmLayers[i].gradPrevOutput = dec.lstmLayers[i].userGradPrevOutput
        else
            enc.lstmLayers[i].userNextGradCell = nn.rnn.recursiveCopy(enc.lstmLayers[i].userNextGradCell, dec.lstmLayers[i].userGradPrevCell)
            enc.lstmLayers[i].gradPrevOutput = nn.rnn.recursiveCopy(enc.lstmLayers[i].gradPrevOutput, dec.lstmLayers[i].userGradPrevOutput)
        end
    end
end
