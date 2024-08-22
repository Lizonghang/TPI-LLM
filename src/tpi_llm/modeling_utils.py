import os
import json
import logging
from transformers.utils import ContextManagers
from transformers.modeling_utils import PreTrainedModel, no_init_weights
from transformers.generation import GenerationConfig
from .generation.utils import TPIGenerationMixin
from .utils import CONFIG_NAME

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_model_config(model_path):
    """
    Loads the model configuration from a JSON file.

    Parameters:
    - model_path: The path to the directory where the model files are stored.

    Returns:
    - config: A dictionary containing the model configuration.
    """
    try:
        config_file = os.path.join(model_path, CONFIG_NAME)
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_file} not found")


class TPIPreTrainedModel(PreTrainedModel, TPIGenerationMixin):
    """
    Base class for all Tensor Parallelism Inference (TPI) models. This class inherits from `PreTrainedModel`
    and `TPIGenerationMixin` to provide a foundation for TPI models that use tensor parallelism during inference.
    """

    @classmethod
    def from_pretrained(cls, model_path, *model_args, **kwargs) -> "TPIPreTrainedModel":
        """
        Instantiate a pretrained TPI model from a pre-trained model configuration.
        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).

        Parameters:
        - model_path: The path to the directory containing the pretrained model weights and configuration.
        - model_args: All remaining positional arguments will be passed to the underlying model's `__init__` method.
        - kwargs: Will be passed to the configuration class initialization function.

        Returns:
        - TPIPreTrainedModel: An instance of the TPI model which is not initialized.
        """
        # local model configurations.
        config, model_kwargs = cls.config_class.from_pretrained(
            model_path, return_unused_kwargs=True, **kwargs)
        config.name_or_path = model_path

        # create a context manager to prevent weight initialization during model creation.
        init_contexts = [no_init_weights(_enable=True)]
        with ContextManagers(init_contexts):
            model = cls(config, *model_args, **model_kwargs)
        assert model.can_generate()

        # make sure token embedding weights are still tied if needed.
        model.tie_weights()
        # set model in evaluation mode to deactivate DropOut by default.
        model.eval()

        try:
            # load the generation configuration.
            kwargs.pop("args")
            model.generation_config = GenerationConfig.from_pretrained(model_path, **kwargs)
        except OSError:
            logger.warning_once("Generation config file not found, "
                                "using a generation config created from the model config.")

        return model
