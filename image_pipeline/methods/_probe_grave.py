from image_pipeline.core.registry import method
@method(id='PROBEG1', name='ProbeNode', category='gpu_shaders')
def _p(out_dir, seed, params=None):
    return {'image': None}
