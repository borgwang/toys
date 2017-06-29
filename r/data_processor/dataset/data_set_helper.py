from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.registry import Register
from common_proto.train.train_job_pb2 import ResamplingType

# over sampling
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_OS_ADASYN),
                  over_sampling.ADASYN)
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_OS_RANDOM),
                  over_sampling.RandomOverSampler)
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_OS_SMOTE),
                  over_sampling.SMOTE)

# under sampling
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_US_ALL_KNN),
                  under_sampling.AllKNN)
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_US_CC),
                  under_sampling.ClusterCentroids)
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_US_CNN),
                  under_sampling.CondensedNearestNeighbour)
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_US_ENN),
                  under_sampling.EditedNearestNeighbours)
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_US_IHT),
                  under_sampling.InstanceHardnessThreshold)
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_US_NCR),
                  under_sampling.NeighbourhoodCleaningRule)
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_US_NM),
                  under_sampling.NearMiss)
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_US_ONE_SIDED),
                  under_sampling.OneSidedSelection)
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_US_RANDOM),
                  under_sampling.RandomUnderSampler)
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_US_RENN),
                  under_sampling.RepeatedEditedNearestNeighbours)
Register.register("resampler",
                  ResamplingType.Type.Name(ResamplingType.TYPE_US_TOMEK_LINKS),
                  under_sampling.TomekLinks)

class DataSetHelper(object):
    pass
