import pytest

from ytrag.youtube import Transcript, extract_video_id, format_ts, video_url

ID = "B5j3uNBH8X4"


@pytest.mark.parametrize(
    "value",
    [
        ID,
        f"https://www.youtube.com/watch?v={ID}",
        f"https://www.youtube.com/watch?v={ID}&t=120s&list=PL123",
        f"https://m.youtube.com/watch?v={ID}",
        f"youtube.com/watch?v={ID}",
        f"https://youtu.be/{ID}",
        f"https://youtu.be/{ID}?t=42",
        f"https://www.youtube.com/shorts/{ID}",
        f"https://www.youtube.com/embed/{ID}",
        f"https://www.youtube.com/live/{ID}?feature=share",
        f"  {ID}  ",
    ],
)
def test_extract_video_id(value):
    assert extract_video_id(value) == ID


@pytest.mark.parametrize(
    "value", ["", "hello", "https://vimeo.com/123456", "https://www.youtube.com/watch?v=short", "https://www.youtube.com/"]
)
def test_extract_video_id_rejects_bad_input(value):
    with pytest.raises(ValueError):
        extract_video_id(value)


def test_format_ts_and_url():
    assert format_ts(0) == "00:00"
    assert format_ts(754.9) == "12:34"
    assert format_ts(3725) == "1:02:05"
    assert video_url(ID, 754.9) == f"https://youtu.be/{ID}?t=754"


def test_transcript_round_trip(transcript):
    restored = Transcript.from_dict(transcript.to_dict())
    assert restored == transcript
    assert restored.duration == pytest.approx(transcript.segments[-1].start + 9.5)
