require 'image'
local log = loadfile('log.lua')()

function getdataSeqHko()
   -- local data = torch.DiskFile(datafile,'r'):readObject()
   -- data size (totalInstances or nsamples=2000?, nSequence_length=20, 1, 64, 64)
    local datasetSeq ={}
   --------------- configuration: -----------------

    log.trace('opening listFile: ', opt.listFile)
    local f = io.open(opt.listFile, 'r')

    local fileList = {}    
    for line in f:lines() do
        table.insert(fileList, line)
    end

    log.info("[init] read data into fileList table, size in total: ", table.getn(fileList))
    log.info("[init] size of input batch ", opt.batchSize, opt.nSeq, opt.imageDepth, opt.imageHeight, opt.imageWidth)

    function datasetSeq:size()
        return table.getn(fileList)
    end


    local ind = 1
    function datasetSeq:SelectSeq()
        log.trace("SelectSeq")

        local inputBatch = torch.Tensor(opt.batchSize, opt.nSeq, opt.imageDepth, opt.imageHeight, opt.imageWidth)
        ind = ind + 1

        for batch_ind = 1, opt.batchSize do -- filling one batch one by one
            log.trace("<selecting> batch #", batch_ind)
            -- math.ceil(torch.uniform(1e-12, nsamples)) 
            -- choose an index in range{1,.. nsamples}
            -- image index
           -- read the 20 frames starting from i
            for frames_id = 1, opt.totalSeqLen do
               local imageName = opt.dataPath..fileList[ (batch_ind - 1) * opt.totalSeqLen + frames_id]
               inputBatch[batch_ind][frames_id] = image.load(imageName)
               log.trace("loading img: ", imageName)
            end
         end
         return inputBatch, ind
      end

   dataSample = torch.Tensor(opt.batchSize, opt.nSeq, opt.imageDepth, opt.imageHeight, opt.imageWidth)
   
   setmetatable(datasetSeq, {__index = function(self, index)
                                       local sample, i = self:SelectSeq()
                                       dataSample:copy(sample)
                                       return {dataSample, i}
                                    end})
   return datasetSeq
end

