import evfuncs
import vocalpy as voc


class TestSequenceDataset:
    def test_round_trip(self, audio_dir_cbin, tmp_path):
        """Test that we can "round trip" a SequenceDataset:
         save it, then load it, and have them be equal"""
        cbin_paths = voc.paths.from_dir(audio_dir_cbin, 'cbin')
        audios = [voc.Audio.read(cbin_path) for cbin_path in cbin_paths]

        segment_params = {
            'threshold': 1500,
            'min_dur': 0.01,
            'min_silent_dur': 0.006,
        }
        segmenter = voc.Segmenter(
            callback=voc.segment.energy,
            segment_params=segment_params,
        )

        seqs = segmenter.segment(audios)

        seq_dataset = voc.dataset.SequenceDataset(sequences=seqs)

        seq_dataset.to_sqlite(db_name='gy6or6-032312.db', dst=tmp_path, replace=True)

        other_seq_dataset = voc.dataset.SequenceDataset.from_sqlite(db_name='gy6or6-032312.db', dst=tmp_path)

        assert other_seq_dataset == seq_dataset
