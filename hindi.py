"""The Hindi Entities Dataset."""
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """"""

_DESCRIPTION = """Hindi Dataset converted for token classification taken from IJNLP 2008."""

_URL = "Datasets/Hindi/"
_TRAINING_FILE = "hindi_train.txt"
_DEV_FILE = "hindi_validation.txt"
_TEST_FILE = "hindi_test.txt"


class PUNJABIConfig(datasets.BuilderConfig):
    """The Hindi Entities Dataset."""

    def __init__(self, **kwargs):
        """BuilderConfig for HINDI.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PUNJABIConfig, self).__init__(**kwargs)


class Hindi(datasets.GeneratorBasedBuilder):
    """The Hindi Entities Dataset."""

    BUILDER_CONFIGS = [
        PUNJABIConfig(
            name="Hindi", version=datasets.Version("1.0.0"), description="The Hindi Entities Dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=['LOCATION', 'BRAND', 'TITLE_OBJECT', 'PERSON', 'DESIGNATION', 'ORGANIZATION', 'ABBREVIATION', 'TIME', 'NUMBER', 'MEASURE', 'TERMS', 'O']
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            current_tokens = []
            current_labels = []
            sentence_counter = 0
            for row in f:
                row = row.rstrip()
                if row:
                    token, label = row.split("\t")
                    current_tokens.append(token)
                    current_labels.append(label)
                else:
                    # New sentence
                    if not current_tokens:
                        # Consecutive empty lines will cause empty sentences
                        continue
                    assert len(current_tokens) == len(current_labels), "💔 between len of tokens & labels"
                    sentence = (
                        sentence_counter,
                        {
                            "id": str(sentence_counter),
                            "tokens": current_tokens,
                            "ner_tags": current_labels,
                        },
                    )
                    sentence_counter += 1
                    current_tokens = []
                    current_labels = []
                    yield sentence
            # Don't forget last sentence in dataset 🧐
            if current_tokens:
                yield sentence_counter, {
                    "id": str(sentence_counter),
                    "tokens": current_tokens,
                    "ner_tags": current_labels,
                }

