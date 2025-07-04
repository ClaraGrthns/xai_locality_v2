from src.explanation_methods.gradient import (IntegratedGradientsHandler,
                                              GuidedBackpropHandler, 
                                              GuidedGradCamHandler,  
                                              DeconvHandler,
                                              SaliencyHandler,
                                              SmoothGradHandler)
from src.explanation_methods.lime import LimeHandler
from src.explanation_methods.lime_captum import LimeCaptumHandler
class ExplanationMethodHandlerFactory:
    METHOD_HANDLERS = {
        "IG": IntegratedGradientsHandler,
        "GuidedBackprob": GuidedBackpropHandler,
        "Deconv": DeconvHandler,
        "GuidedGradCam": GuidedGradCamHandler,
        "Saliency": SaliencyHandler,
        "IG+SmoothGrad": SmoothGradHandler,
        "lime": LimeHandler,
        "lime_captum": LimeCaptumHandler, 
    }

    @staticmethod
    def get_handler(method):
        if method not in ExplanationMethodHandlerFactory.METHOD_HANDLERS:
            raise ValueError(f"Unsupported explanation method: {method}")
        return ExplanationMethodHandlerFactory.METHOD_HANDLERS[method]
