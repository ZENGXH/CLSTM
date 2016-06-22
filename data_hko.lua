require 'image'
local data_verbose = false
dofile 'data_configure.lua'



function getdataSeqHko()
   -- local data = torch.DiskFile(datafile,'r'):readObject()
   -- data size (totalInstances or nsamples=2000?, nSequence_length=20, 1, 64, 64)
    local datasetSeq ={}

   --------------- configuration: -----------------
    local nSamples = opt.nSamples -- 2037 * 4-- data:size(1)
    local nSeq  = opt.nSeq --20 -- data:size(2)
    local nBatch = opt.nBatch
   -- log.INFO (nsamples .. ' ' .. nSeq .. ' ' .. nRows .. ' ' .. nCols)

   ------------- read the powerful txt file! ------
    local f = io.open(opt.listFile, 'r')
    local id = 1
    local fileList = {}
    
    for line in f:lines() do
        fileList[id] = line
        id = id + 1
    end

    assert(table.getn(opt.dataList) == nSeq * nSamples)

    function datasetSeq:size()
        return nsamples
    end

    local ind = 1

    function datasetSeq:SelectSeq()
        log.trace("SelectSeq")
        local input_batch = torch.Tensor(nBatch, nSeq, 
                            opt.imageDepth, opt.imageHeight, opt.imageWidth)

        for batch_ind = 1, nBatch do -- filling one batch one by one
            local i = ind
            ind = ind + 1
            log.trace("<selecting> batch_ind # " + ind)
            -- math.ceil(torch.uniform(1e-12, nsamples)) 
            -- choose an index in range{1,.. nsamples}
            -- image index
            -- read the 20 frames starting from i
            for k = 1, nSeq do
               input_batch[batch_ind][k] = image.load(opt.dataPath..fileList[(i-1) * nSeq + k])

            end
         end
         return input_batch,i
      end

   dataSample = torch.Tensor(nBatch, nSeq, opt.imageDepth, opt.imageHeight, opt.imageWidth)
   
   setmetatable(datasetSeq, {__index = function(self, index)
                                       local sample, i = self:selectSeq()
                                       dataSample:copy(sample)
                                       return {dsample,i}
                                    end})
   return datasetSeq
end

