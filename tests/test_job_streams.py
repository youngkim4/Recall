"""The stream that sends what a job prints to the log of that job."""
import io

import ui_server


def test_what_a_job_prints_goes_to_its_writer_and_the_rest_to_the_stream():
    stream = io.StringIO()
    router = ui_server._JobStreamRouter(stream)

    router.write("no job is running\n")

    assert stream.getvalue() == "no job is running\n"


def test_a_router_whose_stream_is_closed_ends_without_an_error():
    # At the end of a process the stream can close before the router does.
    stream = io.TextIOWrapper(io.BytesIO())
    router = ui_server._JobStreamRouter(stream)
    stream.close()

    router.flush()
    router.close()

    assert router.closed
