from ndsl import Backend, OptimizationConfig


def get_optimization_config(backend: Backend) -> OptimizationConfig:
    if backend.is_gpu_backend():
        return OptimizationConfig(
            stree=OptimizationConfig.Tree(
                enabled=True,
                kernalize=True,
                merger=OptimizationConfig.Tree.Merger(enabled=False, overcompute=False),
            ),
            gpu=OptimizationConfig.GPU(common_gpu_xforms=False),
        )
    else:
        return OptimizationConfig(
            stree=OptimizationConfig.Tree(
                enabled=True,
                kernalize=False,
                merger=OptimizationConfig.Tree.Merger(enabled=True, overcompute=False),
            ),
        )
