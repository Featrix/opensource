from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
import uuid
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd

top = Path(__file__).parent.parent
data_files = Path(__file__).parent / "data"

try:
    import featrixclient as ft
except ImportError:
    sys.path.insert(0, str(top))
    print(f"path is {sys.path}")
    import featrixclient as ft


def get_client(env, client_id, client_secret):
    if env == "dev":
        os.environ['FEATRIX_CLIENT_ID'] = 'bd5ec45d-1c22-49fb-9b14-b3b13b428c68' if client_id is None else client_id
        os.environ['FEATRIX_CLIENT_SECRET'] = \
            '4e7cddd7-dbdc-4f90-9a71-4b54cdec754e' if client_secret is None else client_secret
        target_api_server = "http://localhost:3001"
        allow_unencrypted_http = True
    elif env in ["stage", "prod"]:
        target_api_server = "https://stage.featrix.com" if env == "stage" else "https://app.featrix.com"
        allow_unencrypted_http = False
        if client_id is not None:
            os.environ['FEATRIX_CLIENT_ID'] = client_id
        if client_secret is not None:
            os.environ['FEATRIX_CLIENT_SECRET'] = client_secret
    else:
        raise RuntimeError(f"Unknown environment {env}")

    fc = ft.networkclient.new_client(
        target_api_server,
        allow_unencrypted_http=allow_unencrypted_http,  # DEBUG
    )
    return fc


def wait_for_upload(up, pause=2):
    while up.ready_for_training is False:
        print(f"...waiting for post processing on {up.filename}")
        time.sleep(pause)
        up = up.by_id(up.id, up.fc)
    return up


def wait_for_project(project, pause=None):
    project = project.fc.get_project_by_id(str(project.id))
    if pause is None:
        project.ready(wait_for_completion=True)
        project = project.by_id(project.id, project.fc)
    while project.ready() is False:
        print(f"...waiting for project {project.name} to be ready...")
        time.sleep(pause)
        project = project.by_id(project, project.fc)

    return project


def test_uploads(fc):
    start = datetime.utcnow()
    print("Testing uploads...")
    boston = data_files / "2000-boston.csv"
    if not boston.exists():
        raise RuntimeError("Data file 2000-boston.csv does not exist")
    up = fc.upload_file(boston)
    up = wait_for_upload(up)
    print(f"...uploaded {up.filename} successfully in {(datetime.utcnow() - start).total_seconds()}")

    start = datetime.utcnow()
    boston2 = data_files / "8000-boston.csv"
    df = featrix_wrap_pd_read_csv(boston2)
    up = fc.upload_file(df, label="8000-boston.csv")
    up = wait_for_upload(up)
    print(f"...uploaded {up.filename} successfully in {(datetime.utcnow() - start).total_seconds()}")

    start = datetime.utcnow()
    p = fc.create_project(f"Upload testing {uuid.uuid4()}")
    colors = data_files / "colors.csv"
    up = fc.upload_file(colors, associate=True)
    up = wait_for_upload(up)
    if fc.current_project.ready() is False:
        raise RuntimeError(f"Upload was ready but project p1 {p.name} was not")
    print(f"...uploaded {up.filename} successfully in {(datetime.utcnow() - start).total_seconds()}")

    start = datetime.utcnow()
    p2 = fc.create_project(f"Upload testing {uuid.uuid4()}")
    wh = data_files / "weight-height.csv"
    up = fc.upload_file(wh, associate=p2)
    up = wait_for_upload(up)
    fc.current_project = str(p2.id)
    if fc.current_project.ready() is False:
        raise RuntimeError(f"Upload was ready but project p2 {p2.name} was not")
    print(f"...uploaded {up.filename} successfully in {(datetime.utcnow() - start).total_seconds()}")

    start = datetime.utcnow()
    p3 = fc.create_project(f"Upload testing {uuid.uuid4()}")
    boston3 = data_files / "merged-boston-fire-250.csv"
    df = featrix_wrap_pd_read_csv(boston3)
    up = fc.upload_file(df, label=boston3.name, associate=True)
    up = wait_for_upload(up)
    fc.current_project = str(p3.id)
    if fc.current_project.ready() is False:
        raise RuntimeError(f"Upload was ready but project p3 {p3.name} was not")
    print(f"...uploaded {up.filename} successfully in {(datetime.utcnow() - start).total_seconds()}")


def test_nf(fc):
    animals_small = data_files / "animals-1k.csv"
    animals_small2 = data_files / "animals-1kv2.csv"
    animals_small3 = data_files / "animals-1kv3.csv"

    # 1 - create NF with explicit project and upload
    start = datetime.utcnow()
    print("NF Test 1/3...Testing creating NF with explicit project/upload (this will take few minutes)")
    p = fc.create_project(f"NF test 1 - {uuid.uuid4()}")
    fc.upload_file(animals_small, associate=p)
    # We need to refresh project -- it's in the fc so we can set it as the current or just grab the new one from cache
    p = wait_for_project(p)
    print(f"......creating nf in project {p.name}")
    nf, job, job = fc.create_neural_function(target_fields="Animal", project=p, wait_for_completion=True)
    nf.predict({'weight': 22})
    print(f"......created nf {nf.name} and ran prediction in in {(datetime.utcnow() - start).total_seconds()} secs")

    # 2 - create NF with implicit project and upload
    start = datetime.utcnow()
    print("NF Test 2/3...Testing creating NF with implicit project/upload (this will take few minutes)")
    p = fc.create_project(f"NF test 2 - {uuid.uuid4()}")
    fc.upload_file(animals_small2, associate=p)
    p = wait_for_project(p)
    print(f"......creating nf in project {p.name}")
    nf2, job, job = fc.create_neural_function(target_fields="Animal", wait_for_completion=True)
    nf2.predict({'weight': 15})
    print(f"......created nf {nf2.name} and ran prediction in in {(datetime.utcnow() - start).total_seconds()} secs")

    # 3 create NF in one pass (hand in upload and project
    start = datetime.utcnow()
    print("NF Test 3/3...Testing creating NF with single call including project "
          "creation/upload data (this will take few minutes)")
    nf3, job, job = fc.create_neural_function(target_fields="weight", project=f"NF Test 3 - {uuid.uuid4}",
                                              files=[animals_small3], wait_for_completion=True)
    nf3.predict({'Animal': "Dog"})
    print(f"......created nf {nf3.name} and ran prediction in in {(datetime.utcnow() - start).total_seconds()} secs")

    return


def test_es(fc):
    animals_small = data_files / "animals-1k.csv"
    animals_small4 = data_files / "animals-1kv4.csv"

    # 1 - create NF with explicit project and upload
    print("ES Test 1/3...Testing creating ES with explicit project/upload (this will take few minutes)")
    start = datetime.utcnow()
    p = fc.create_project(f"ES test 1 - {uuid.uuid4()}")
    # This will test using a data file we already uploaded as well
    fc.upload_file(animals_small, associate=p)
    p = wait_for_project(p)
    es, job = fc.create_embedding_space(project=p, name=f"ES in {p.name}", wait_for_completion=True)
    assert job.finished is True, f"Job {job.id} did not finish with wait_for_completion"
    assert job.error is False, f"Job {job.id} failed"
    print(f"......created es {es.name} in {(datetime.utcnow() - start).total_seconds()} secs")

    # 2 - create NF with implicit project and upload
    print("ES Test 2/3...Testing creating ES with implicit project/upload (this will take few minutes)")
    start = datetime.utcnow()
    p = fc.create_project(f"ES test 2 - {uuid.uuid4()}")
    fc.upload_file(animals_small4, associate=p)
    wait_for_project(p)
    es2, job = fc.create_embedding_space(wait_for_completion=True)
    assert job.finished is True, f"Job {job.id} did not finish with wait_for_completion"
    assert job.error is False, f"Job {job.id} failed"
    print(f"......created es {es2.name} in {(datetime.utcnow() - start).total_seconds()} secs")

    # 3 create NF in one pass (hand in upload and project
    print("ES Test 3/3...Testing creating ES with single call including project "
          "creation/upload data (this will take few minutes)")
    start = datetime.utcnow()
    es3, job = fc.create_embedding_space(project=f"ES Testing {uuid.uuid4}", name="ES test",
                                         files=[animals_small4], wait_for_completion=True)
    assert job.finished is True, f"Job {job.id} did not finish with wait_for_completion"
    assert job.error is False, f"Job {job.id} failed"
    print(f"......created es {es3.name} in {(datetime.utcnow() - start).total_seconds()} secs")
    return


def _find_bad_line_number(file_path: Path | str = None, buffer: bytes | str = None):
    try:
        if file_path:
            buffer = file_path.read_text()

        reader = csv.reader(buffer)
        linenumber = 1
        try:
            for _ in reader:
                linenumber += 1
        except Exception as e:  # noqa
            return linenumber
    except:  # noqa
        pass
    return -1


# A wrapper for dealing with CSV files.
def featrix_wrap_pd_read_csv(
    file_path: str | Path = None, buffer: bytes | str = None, on_bad_lines="skip"
):
    """
    If you want to split CSVs in your notebook and so on when working
    with Featrix, this function should be used to capture the extra work
    around pandas' `pd.read_csv` that you'll want for best performance
    with Featrix. We will add split and a way to get back the test df
    to the client in a future release.

    Any column with an 'int' type -- meaning there doesn't seem to be a
    header line in the CSV -- will be renamed to `column_N`.

    Parameters
    ----------
    file_path : str
        Path to the CSV on your local system.
    buffer: str or bytes
        The CSV already in buffer
    on_bad_lines: str
        What to do with bad lines. By default, we 'skip', but you may want to 'error'.
        This is passed directly to `pd.read_csv`.

    This can raise exceptions if the file is not found or seems to be empty.

    """
    if not file_path and not buffer:
        raise ValueError(
            "No data provided via buffer or path to featrix_wrap_pd_read_csv"
        )
    if file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file {file_path}")
        # get the size of the file
        sz = os.path.getsize(file_path)
        if sz == 0:
            raise Exception(f"The file {file_path} appears to be 0 bytes long.")
    elif isinstance(buffer, bytes):
        buffer = buffer.decode()
    buffer_io = StringIO(buffer) if buffer else None

    sniffer = csv.Sniffer()
    if buffer:
        dialect = sniffer.sniff(buffer)
        has_header = sniffer.has_header(buffer)
    else:
        with open(file_path, newline="", errors='ignore') as csvfile:
            # For some very wide files, 2K isn't enough.
            # It's possible 256K isn't either, but one has to draw the line!
            try:
                sample = csvfile.read(32 * 1024)
            # except UnicodeDecodeError as err:
                # print("bad unicode:",dir(err))
                # print("err.reason: ", err.reason)
                # print("err.start: ", err.start)
                # print("err.end: ", err.end)
            except:  # noqa
                bad_line = _find_bad_line_number(file_path=file_path, buffer=buffer)
                if bad_line > 0:
                    print("first BAD LINE WAS ...", bad_line)

            dialect = sniffer.sniff(sample)
            has_header = sniffer.has_header(sample)

    csv_parameters = {
        'delimiter': dialect.delimiter,
        'quotechar': dialect.quotechar,
        'escapechar': dialect.escapechar,
        'doublequote': dialect.doublequote,
        'skipinitialspace': dialect.skipinitialspace,
        'quoting': dialect.quoting,
        # Pandas does not support line terminators > 1 but Sniffer returns things like '\r\n'
        # 'lineterminator': dialect.lineterminator
    }

    if has_header:
        try:
            df = pd.read_csv(
                file_path or buffer_io,
                # Pandas doesn't take the same dialect as csv.Sniffer produces so we create csv_parameters
                # dialect=dialect,
                on_bad_lines=on_bad_lines,
                encoding_errors='ignore',
                **csv_parameters
            )
        except csv.Error as err:
            bad_line = _find_bad_line_number(file_path=file_path, buffer=buffer)
            if bad_line > 0:
                print("first BAD LINE WAS ...", bad_line)
            s_err = str(err)
            print(s_err)
            # FIXME: Not sure if there is something we can do if the buffer is hosed?
            if (
                s_err is not None
                and s_err.find("malformed") >= 0
                and file_path is not None
            ):
                df = pd.read_csv(
                    file_path,
                    # Pandas doesn't take the same dialect as csv.Sniffer produces so we create csv_parameters
                    # dialect=dialect,
                    on_bad_lines=on_bad_lines,
                    lineterminator="\n",
                    **csv_parameters
                )
                print("recovered")
            else:
                print("c'est la vie")
                raise err
            # endif

        # if any of the columns have an 'int' type, rename it.
        if df is not None:
            cols = list(df.columns)
            renames = {}
            for idx, c in enumerate(cols):
                if not isinstance(c, str):
                    renames[c] = "column_" + str(c)
            if len(renames) > 0:
                df.rename(columns=renames, inplace=True)

        return df

    if not has_header:
        # Try again -- and see.

        try:
            df = pd.read_csv(file_path or buffer_io,  **csv_parameters)
            cols = df.columns
            if len(cols) >= 0:
                if cols[0].startswith("Unnamed"):
                    # still no good.
                    raise Exception(
                        f"CSV file {file_path} doesn't seem to have a header line, which means it does not "
                        "have labels for the columns. This will make creating predictions on "
                        "specific targets difficult!"
                    )
            return df
        except Exception as err:  # noqa - catch anything
            traceback.print_exc()
            raise Exception(
                f"CSV file {file_path} doesn't seem to have a header line, which means it does not "
                "have labels for the columns. This will make creating predictions on specific targets difficult! [2]"
            )

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--environment", "-e", default="dev", choices=["dev", "prod", "stage"],
                    help="Environment to run the tests on")
    ap.add_argument("--client-id", "-c", help="Run with this client id")
    ap.add_argument("--client-secret", "-s", help="Run with this client secret")

    args = ap.parse_args()
    fc = get_client(args.environment, args.client_id, args.client_secret)

    start = datetime.utcnow()
    test_uploads(fc)
    print(f"Finished Upload testing in {(datetime.utcnow() - start).total_seconds()} seconds.")
    substart = datetime.utcnow()
    test_nf(fc)
    print(f"Finished Neural Function testing in {(datetime.utcnow() - substart).total_seconds()} seconds.")
    substart = datetime.utcnow()
    test_es(fc)
    print(f"Finished Embedding Space testing in {(datetime.utcnow() - substart).total_seconds()} seconds.")
    print(f"Smoke test complete in {(datetime.utcnow() - start).total_seconds()} seconds.")


if __name__ == "__main__":
    main()
