from __future__ import annotations

import json
import pathlib

import attrs
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ..audio import Audio
from ..sequence import Sequence
from ..unit import Unit
from . import schema


@attrs.define
class SequenceDataset:
    """A dataset of sequences of units
    as used in acoustic communication research.

    The sequences of units are segmented from audio data,
    then used for further analysis and/or annotation.

    Attributes
    ----------
    sequences: list
        A :class:`list` of :class:`vocalpy.Sequence` instances.
    segment_params: dict or list of dict
        Parameters used to segment;
        keyword arguments to callback used with :class:`vocalpy.Segmenter`.

    Methods
    -------
    to_sqlite
    from_sqlite
    """

    sequences: list[Sequence] = attrs.field()

    @sequences.validator
    def is_list_of_seqs(self, attribute, value):
        if not isinstance(value, list):
            raise TypeError(f"`sequences` must be a list but type was: {type(value)}")
        if not all([isinstance(element, Sequence) for element in value]):
            raise TypeError("All elements in `sequences` list must be of type `vocalpy.Sequence`.")

    segment_params: dict | list[dict] = attrs.field(init=False)

    def __attrs_post_init__(self):
        uniq_segment_params = []
        # need to do it this way since we can't hash a dict to use `set` to find unique
        # https://stackoverflow.com/questions/11092511/list-of-unique-dictionaries
        for seq in self.sequences:
            if seq.segment_params not in uniq_segment_params:
                uniq_segment_params.append(seq.segment_params)

        if len(uniq_segment_params) == 1:
            self.segment_params = uniq_segment_params[0]
        else:  # if there's more than one set
            self.segment_params = uniq_segment_params

    def to_sqlite(self, db_name=":memory:", dst=None, replace=False, echo=False):
        """Save this :class:`vocalpy.dataset.SequenceDataset` to a SQLite database file.

        This method creates a new database file every time it is called.
        It will not overwrite existing files, unless the `replace` argument
        is set to `True`.

        Parameters
        ----------
        db_name : str
            Name of file that database should be saved to.
        dst : str, pathlib.Path, optional.
            Destination, a directory. If not specified,
            the database will be saved to the current working directory.
        replace : bool
            If True, replace any existing file.
        echo : bool
            If True, log SQL statements when building database.
        """
        if dst:
            dst = pathlib.Path(dst)
            if not dst.exists() or not dst.is_dir():
                raise NotADirectoryError(f"`dst` not found or not recognized as a directory: {dst}")
        else:
            dst = pathlib.Path(".")
        url = dst / db_name

        if url.exists():
            if not replace:
                raise FileExistsError(
                    f"SQLite database file already exists: {url}. "
                    f"Either use a different name, or call `to_sqlite` with `replace=True`, "
                    f"which will overwrite the existing database file."
                )
            url.unlink()

        engine = create_engine(f"sqlite:///{url}", echo=echo)

        schema.sequence.SequenceDatasetBase.metadata.create_all(engine)

        # TODO: handle checking here --
        # actually make sure in Segmenter that `segment_params` can be serialized
        orm_segment_params = []
        if isinstance(self.segment_params, dict):
            uniq_segment_params = [self.segment_params]
        else:
            uniq_segment_params = self.segment_params

        for ind, segment_params_dict in enumerate(uniq_segment_params):
            segment_params_fname = f"{db_name}-segment-params-{ind + 1}.json"
            segment_params_json_path = dst / segment_params_fname
            with segment_params_json_path.open("w") as fp:
                json.dump(segment_params_dict, fp)
            orm_segment_params.append(schema.sequence.SegmentParams(path=str(segment_params_json_path)))

        with Session(engine) as session, session.begin():
            for an_orm_segment_params in orm_segment_params:
                session.add(an_orm_segment_params)

            for seq in self.sequences:
                audio = schema.sequence.Audio(path=str(seq.audio.path))
                session.add(audio)

                # make and add sequence, referring to audio and segment params
                ind = uniq_segment_params.index(seq.segment_params)
                an_orm_segment_params = orm_segment_params[ind]

                sequence = schema.sequence.Sequence(
                    audio=audio,
                    segment_params=an_orm_segment_params,
                    onset=seq.onset,
                    offset=seq.offset,
                    method=seq.method,
                )
                session.add(sequence)

                # make and add units
                for seq_unit in seq.units:
                    unit = schema.sequence.Unit(
                        onset=seq_unit.onset, offset=seq_unit.offset, label=seq_unit.label, sequence=sequence
                    )
                    session.add(unit)
            # ---- implicit session.commit() when we __exit__ context + begin() from above

    @classmethod
    def from_sqlite(cls, db_name: str, dst=None, echo=False):
        """Load a :class:`vocalpy.dataset.SequenceDatset` from a SQLite database file.

        Parameters
        ----------
        db_name : str
            Name of file that database should be loaded from.
        dst : str, pathlib.Path, optional
            Destination, a directory. If not specified,
            defaults to the current working directory.
        echo : bool
            If True, log SQL statements when querying the database
            to load the dataset.

        Returns
        -------
        seq_dataset : vocalpy.dataset.SequenceDataset
        """
        if dst:
            dst = pathlib.Path(dst)
            if not dst.exists() or not dst.is_dir():
                raise NotADirectoryError(f"`dst` not found or not recognized as a directory: {dst}")
        else:
            dst = pathlib.Path(".")
        url = dst / db_name

        if not url.exists():
            raise FileNotFoundError(
                f"`db_name` not found at `dst`: {url}. Please check values for both `db_name` and `dst`."
                f"Note that `dst` defaults to the current working directory if not specified."
            )

        url = f"sqlite:///{url}"

        engine = create_engine(url, echo=echo)

        schema.sequence.SequenceDatasetBase.metadata.create_all(engine)

        with Session(engine) as session, session.begin():
            seg_params_stmt = select(schema.sequence.SegmentParams).order_by(schema.sequence.SegmentParams.id)
            seg_params_result = session.scalars(seg_params_stmt).all()

            segment_params = []
            for model_segment_params in seg_params_result:
                path = model_segment_params.path
                with pathlib.Path(path).open("r") as fp:
                    this_seg_params = json.load(fp)
                segment_params.append(this_seg_params)

            audios = []
            audio_stmt = select(schema.sequence.Audio).order_by(schema.sequence.Audio.id)
            audio_result = session.scalars(audio_stmt).all()
            for model_audio in audio_result:
                audios.append(Audio(path=model_audio.path))

            seqs_stmt = select(schema.sequence.Sequence).order_by(schema.sequence.Sequence.id)
            seqs_result = session.scalars(seqs_stmt).all()

            seqs = []
            for model_sequence in seqs_result:
                units_stmt = (
                    select(schema.sequence.Unit)
                    .where(schema.sequence.Unit.sequence_id == model_sequence.id)
                    .order_by(schema.sequence.Unit.id)
                )
                seq_units_result = session.scalars(units_stmt).all()

                units = []
                for model_unit in seq_units_result:
                    unit = Unit(onset=model_unit.onset, offset=model_unit.offset, label=model_unit.label)
                    units.append(unit)

                seq = Sequence(
                    units=units,
                    audio=audios[model_sequence.audio_id - 1],
                    method=model_sequence.method,
                    segment_params=segment_params[model_sequence.segment_params_id - 1],
                )
                seqs.append(seq)

        return cls(sequences=seqs)
