from pathlib import Path
from typing import Dict, List, Optional, Sequence, Iterable, Union
import itertools
import logging
import warnings

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from pandas import read_csv
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

logger = logging.getLogger(__name__)


@DatasetReader.register("references")
class ReferencesDatasetReader(DatasetReader):
    """
    Reads instances from a csv file with the following columns:
    ```
    case_number: The case number of the lawsuit
    procedure: The stage of the law suit: 一审、二审、再审
    case_number_first_trial: If it is the second trial, then there is also the case number of the first trial.
    date_of_trial: The date when the lawsuit ends.
    case_sector: The sector the lawsuit as defined by the Supreme People's Court
    province: The province of the lawsuit.
    court_name: The court that holds the lawsuit.
    actor_for_judgement: The actor that are present in the lawsuit：上诉人（原告）、上诉人（被告）、被上诉人（原告）、被上诉人（被告）
    actor_for_case: The actor that are present in the case as specified in the first trial：原告、被告、法官、第三人
    支持起诉机关.
        Should be used for identifying references from different actors.
    name_of_the_actor: The name of the actor. Here the notation | means that the actor is 被告 in the first trial,
        but does not appeal.
    actor_category: The category of the actor in the case: government, ngo, procuratorate, enterprise, individual,
        and the court.
    state_or_civil_society: The category of the actor in the 环境公益诉讼案例. Not very relevant.
    references_all: All the references from the different actor_for_case. Here all references are those that are in the
        bracket 《》. However, sometimes some references are abbreviated and without the bracket 《》.
        In a few cases there are no bracket for all references. Lots of those references are not law at all,
        including for instance，合同、报告、涵、证明、供词、鉴定书、说明、答复、专家意见、许可证、资格证、意见书、建议书、etc.
        For the moment Wen and Yi are not able to list all of them and have to count on human judgement
    reference_law_all: References include both soft and hard law.
    reference_soft_law: All the soft law references, including 标准。Wen and Yi are not able to define an operationable
        soft law concept, because there are too many forms of soft law in China, including 的意见，的通知，方案，计划，规范
        推荐名录，推荐办法.，推荐方法, etc. Basically, soft law also applies to a broad audience but is not legally binding.
        http://search.chinalaw.gov.cn/search2.html has the most comprehensive collection of binding laws in China,
        but it does not work well for the moment.
    reference_soft_law_without_standard: All the soft law references, excluding 标准。
    coder: The name of the coder.
    full_text: Full text from the trial.
    case_handling_fee_text: The paragraph on the case handling fee. May not be very relevant at the moment.
    note: Interesting note.
    ```
    Here the notation | means that the references are in the first trial or the second trial, because during the second
    trial, it often includes the content from the first trial. Thus for all cases, I tried to find the seond trial legal
    judgements, only in a very few cases I failed. The notation {} means that these references are used as evidence or
    证据, instead of arguments. For the judge or 法官, / means before this, all the references are used as facts in the
    case 确认事实, after this and before \, those references are used as judge‘s arguments 本院认为，and after \ all the
    references are used as the final judgments 判决如下.

    and converts it into a `Dataset` suitable for sequence tagging.
    Each `Instance` contains the words in the `"tokens"` `TextField`.
    Values will get loaded into the `"tags"` `SequenceLabelField`.
    This dataset reader simply treats each sentence as an independent `Instance`.
    Registered as a `DatasetReader` with name "references".
    # Parameters
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    coding_scheme : `str`, optional (default=`IOB1`)
        Specifies the coding scheme for tags.
        In the IOB1 scheme, I is a token inside a span, O is a token outside
        a span and B is the beginning of span immediately following another
        span of the same type.
    label_namespace : `str`, optional (default=`labels`)
        Specifies the namespace for the tags.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        coding_scheme: Optional[str] = "IOB1",
        label_namespace: str = "labels",

        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if coding_scheme not in ("IOB1", "BIOUL"):
            raise ConfigurationError(
                "unknown coding_scheme: {}".format(coding_scheme)
            )

        self.coding_scheme = coding_scheme
        self.label_namespace = label_namespace
        self._original_coding_scheme = "IOB1"
        self._spacy_model = "zh_core_web_sm"
        self._tokenizer = SpacyTokenizer(self._spacy_model)
        self._sentence_segmenter = SpacySentenceSplitter(self._spacy_model)

    @overrides
    def _read(self, file_path: Union[Path, str]) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading instances from lines in file at: %s", file_path)
        df = read_csv(file_path)

        for line in df.itertuples():
            refs = str(line.references_all).strip("；").replace("|", "").split("；")
            for sentence in self._sentence_segmenter.split_sentences(line.full_text):
                tokens = self._tokenizer.tokenize(sentence)
                tags = ["O" for _ in tokens]
                for i in range(len(tokens)):
                    for j in range(i + 1, len(tokens) + 1):
                        if "".join(token.text for token in tokens[i:j]) in refs:
                            tags[i] = "B"
                            tags[i + 1:j] = ["I"] * (j - i - 1)
                yield self.text_to_instance(tokens, tags)

    def text_to_instance(  # type: ignore
        self,
        tokens: List[Token],
        tags: List[str] = None,
    ) -> Instance:

        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence,
                                             "metadata": MetadataField({"words": [x.text for x in tokens]})}

        # Recode the labels if necessary.
        if self.coding_scheme == "BIOUL":
            tags = to_bioul(tags, encoding=self._original_coding_scheme)

        # Add tag label to instance
        instance_fields["tags"] = SequenceLabelField(tags, sequence, self.label_namespace)

        return Instance(instance_fields)
