#  -*- coding: utf-8 -*-
#############################################################################
#
#  Copyright (c) 2024, Featrix, Inc.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#############################################################################
#
#     Welcome to...
#
#      _______ _______ _______ _______ ______ _______ ___ ___
#     |    ___|    ___|   _   |_     _|   __ \_     _|   |   |
#     |    ___|    ___|       | |   | |      <_|   |_|-     -|
#     |___|   |_______|___|___| |___| |___|__|_______|___|___|
#
#                                                 Let's embed!
#
#############################################################################
#
#  Sign up for Featrix at https://app.featrix.com/
# 
#############################################################################
#
#  Check out the docs -- you can either call the python built-in help()
#  or fire up your browser:
#
#     https://featrix-docs.readthedocs.io/en/latest/
#
#  You can also join our community Slack:
#
#     https://bits.featrix.com/slack
#
#  We'd love to hear from you: bugs, features, questions -- send them along!
#
#     hello@featrix.ai
#
#############################################################################
#
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import List

import pandas as pd


top = Path(__file__).parent.parent
uploads_to_delete = []
projects_to_delete = []


def generate_data_file(input_file, cnt, column_name=None, output_file=None):
    df = pd.read_csv(input_file)
    if cnt > len(df):
        cnt = len(df)
    if column_name:
        out = df.groupby(column_name).sample(cnt // 2)
    else:
        out = df.sample(cnt)
    if output_file:
        out.to_csv(output_file, index=False)
    else:
        return out


def get_client(featrixhost_env):
    import featrixclient as fc

    try:
        client_id = os.environ["FEATRIX_CLIENT_ID"]
        client_secret = os.environ["FEATRIX_CLIENT_SECRET"]
    except:
        print("FEATRIX_CLIENT_ID and FEATRIX_CLIENT_SECRET must be set in the environment")
        exit(2)

    target_api_server = "https://app.featrix.com"
    allow_unencrypted_http = False

    if featrixhost_env == "dev":
        target_api_server = "http://localhost:3001"
        allow_unencrypted_http = True
    else:
        target_api_server = featrixhost_env

    fc = fc.networkclient.new_client(
        target_api_server,
        client_id=client_id,
        client_secret=client_secret,
        allow_unencrypted_http=allow_unencrypted_http,  # DEBUG
    )
    return fc


def wait_for_upload(up, pause=2):
    while up.ready_for_training is False:
        print(f"...waiting for post processing on {up.filename}")
        time.sleep(pause)
        up = up.by_id(up.id, up._fc)
    return up


def wait_for_project(project, pause=None):
    project = project._fc.get_project_by_id(str(project.id))
    if pause is None:
        project.ready(wait_for_completion=True)
        project = project.by_id(project.id, project._fc)
    while project.ready() is False:
        print(f"...waiting for project {project.name} to be ready...")
        time.sleep(pause)
        project = project.by_id(project, project._fc)

    return project


def test_uploads(
    fc,
    data_dir: Path,
    test_cases: List[Dict],
    verbose: bool = False,
    raise_on_error: bool = False,
):
    from featrixclient.utils import featrix_wrap_pd_read_csv

    start = datetime.utcnow()
    if verbose:
        print("\nTesting uploads\n")

    with tempfile.TemporaryDirectory() as _dir:
        td = Path(_dir)

        for test_idx, test_case in enumerate(test_cases):
            if verbose:
                print(
                    f"Starting Upload test case {test_idx}:\n{json.dumps(test_case, indent=4)}"
                )

            project = None
            try:
                target_file = td / f"uploadtest-{test_idx}-{uuid.uuid4()}.csv"
                if verbose:
                    print(
                        f"...generating data file from {data_dir / test_case['name']}"
                    )
                generate_data_file(
                    data_dir / test_case["name"],
                    test_case.get("sample_size", 1000),
                    output_file=target_file,
                )
                upload_target = (
                    featrix_wrap_pd_read_csv(target_file)
                    if test_case.get("df", False)
                    else target_file
                )
                associate = test_case.get("associate", False)
                if not associate and not test_case.get("project", False):
                    upload = fc.upload_file(target_file)
                    upload = wait_for_upload(upload)
                else:
                    project = fc.create_project(
                        f"Upload test project {test_idx} {uuid.uuid4()}"
                    )
                    if associate:
                        upload = fc.upload_file(upload_target, associate=project)
                    else:
                        upload = fc.upload_file(upload_target, associate=True)
                    upload = wait_for_upload(upload)
                    if project.ready() is False:
                        raise RuntimeError(
                            f"Upload was ready but project {test_idx} {project.name} was not"
                        )

                if verbose:
                    print(
                        f"...uploaded {upload.filename} successfully in "
                        f"{(datetime.utcnow() - start).total_seconds()}"
                    )
                #uploads_to_delete.append(upload)
                #if project:
                #    projects_to_delete.append(project)
            except Exception as e:  # noqa
                print("########### ERROR #########################")
                print(f"Failed Upload test case {test_idx} with error {e}")
                print(traceback.format_exc())
                print("########### ERROR #########################")
                if raise_on_error:
                    raise
    if verbose:
        print(
            f"\nTest Uploads finished in {(datetime.utcnow()-start).total_seconds()}\n"
        )
    return


def test_nf(
    fc,
    data_dir: Path,
    test_cases: List[Dict],
    raise_on_error: bool = True,
):
    print("\nTest Neural Function creation\n")
    with tempfile.TemporaryDirectory() as _dir:
        td = Path(_dir)
        for test_idx, test_case in enumerate(test_cases):
            try:
                print(
                    f"Starting Neural Function test case {test_idx}:\n{json.dumps(test_case, indent=4)}"
                )

                start = datetime.utcnow()
                target_file = td / f"nf_test_{test_idx}-{uuid.uuid4()}.csv"
                project_name = f"NF smoke test {test_idx} - {uuid.uuid4()}"
                # automation = test_case.get("automation", "upload")
                sample_by = test_case.get("sample_by", test_case.get("target"))
                target_column = test_case.get("target", test_case.get("sample_by"))
                if sample_by is None:
                    raise RuntimeError(f"NF Test Case {test_idx} mis-defined")
                generate_data_file(
                    data_dir / test_case["name"],
                    test_case.get("sample_size", 1000),
                    output_file=target_file,
                    column_name=sample_by,
                )
    
                print("------------------------------------------------------------------------------ create 'bad' project")
                bad_project_name = f"NF smoke test 0 - {uuid.uuid4()}"
                project = fc.create_project(bad_project_name)
                # do NOT add data

                # try to train an embedding space.
                from featrixclient.exceptions import FeatrixNotReadyException, FeatrixException

                test_passed = False
                try:
                    bad_es = project.create_embedding_space(wait_for_completion=False)
                except FeatrixNotReadyException:
                    test_passed = True
                
                if not test_passed:
                    raise Exception("create_embedding_space should have failed on an empty project.")
                else:
                    print("PASS")

                if project.ready():
                    raise Exception("project.ready returned TRUE but should have returned FALSE")

                print("------------------------------------------------------------------------------ delete project")
                print("project = ", project)
                print("PASS")

                existing_p1 = fc.get_project_by_name(bad_project_name)
                print("existing_p1 = ", existing_p1)
                assert existing_p1 is not None
                project.delete()
                print("PASS")

                existing_p2 = fc.get_project_by_name(bad_project_name)
                print("existing_p2 = ", existing_p2)
                assert existing_p2 is None
                print("PASS")

                project = fc.create_project(bad_project_name)
                print("project = ", project)
                assert project is not None
                print("PASS")


                print("------------------------------------------------------------------------------ create project")
                project = fc.create_project(f"NF smoke test 1 - {uuid.uuid4()}")
                upload = fc.upload_file(target_file, associate=project)
                project = wait_for_project(project)
                print(f"......creating nf in project {project.name}")
                # Here we will create/train the embedding, then create the neural.

                print("------------------------------------------------------------------------------ create embedding space")
                num_epochs = 50 #20 + test_idx + 1

                es = project.create_embedding_space(
                    name="ES test", 
                    wait_for_completion=True, 
                    epochs=num_epochs
                )

                if es.ready():
                    print("The embedding space is ready, as it should be.")
                else:
                    assert False, "bug in es.ready()"

                assert (es.training_history[0].epochs == num_epochs), "num_epochs doesn't match the ES!"

                print("------------------------------------------------------------------------------ create neural function no_waiting")
                not_ready_nf = es.create_neural_function(
                    target_field=target_column, 
                    wait_for_completion=False,
                    epochs=num_epochs
                )

                # Right now, not_ready_nf is None
                assert not_ready_nf is None, "We will improve this later."
                # was_ready = not_ready_nf.ready()
                # assert was_ready == False, f"was_ready should be False but it is {was_ready}"

                print("------------------------------------------------------------------------------ create neural function")
                nf = es.create_neural_function(
                    target_field=target_column, 
                    wait_for_completion=True,
                    epochs=num_epochs
                )

                if nf.ready():
                    print("The neural function is ready, as it should be.")
                else:
                    assert False, "bug in nf.ready()"
                # FIXME: need negative tests here too??

                print("------------------------------------------------------------------------------ test lambda filter")
                # test lambda filters.
                nfList = es.neural_functions(lambda x: x.training_state == "trained")
                print("trained list = ", len(nfList))
                # print("nf = ", nf)

                
                print("------------------------------------------------------------------------------ negative create neural function")
                had_error = False
                try:
                    broken_nf = es.create_neural_function(
                        target_field="WRONG_FIELD_NOT_IN_DATA", 
                        wait_for_completion=True
                    )
                except FeatrixException:
                    print("got the exception--good")
                    had_error = True
                
                if not had_error:
                    raise Exception("Expected an error when training on a bad field.")

                if "query" in test_case:
                    query = test_case["query"]
                    print("------------------------------------------------------------------------------ prediction test")
                    result = nf.predict(query)
                    print(result)
                    assert len(result) == len(query)
                
                    print("------------------------------------------------------------------------------ prediction test [dataframe]")

                    df_test = pd.DataFrame(query)
                    print(df_test)

                    result2 = nf.predict(df_test)
                    assert len(result2) == len(query)
                else:
                    print("NO QUERY!")

                print("------------------------------------------------------------------------------ create embedding space encoder override")
                num_epochs = 1 #20 + test_idx + 1

                es_encoder_override = project.create_embedding_space(
                    name="ES test encoder override", 
                    wait_for_completion=True, 
                    epochs=num_epochs
                )

                if es.ready():
                    print("The embedding space is ready, as it should be.")
                else:
                    assert False, "bug in es.ready()"

                assert (es.training_history[0].epochs == num_epochs), "num_epochs doesn't match the ES!"




                uploads_to_delete.append(project)
                projects_to_delete.append(upload)
            except Exception as e:  # noqa
                print("########### ERROR #########################")
                print(f"Failed test case {test_idx} with error {e}")
                print(traceback.format_exc())
                print("########### ERROR #########################")
                if raise_on_error:
                    raise
    print(
        f"\nTesting Neural Function creation finished in {(datetime.utcnow()-start).total_seconds()}\n"
    )
    return


def test_es(
    fc,
    data_dir: Path,
    test_cases: List[Dict],
    verbose: bool = False,
    raise_on_error: bool = False,
):
    if verbose:
        print("\nTest Embedding Space creation\n")
    with tempfile.TemporaryDirectory() as _dir:
        td = Path(_dir)

        for test_idx, test_case in enumerate(test_cases):
            try:
                if verbose:
                    print(
                        f"Starting ES test case {test_idx}:\n{json.dumps(test_case, indent=4)}"
                    )
                start = datetime.utcnow()
                target_file = td / f"es_test_{test_idx}-{uuid.uuid4()}.csv"
                project_name = f"ES smoke test {test_idx} - {uuid.uuid4()}"
                automation = test_case.get("automation", "upload")
                sample_by = test_case.get("sample_by")
                if sample_by is None:
                    raise RuntimeError(
                        f"ES Test Case {test_idx} mis-defined, no sample_by"
                    )
                generate_data_file(
                    data_dir / test_case["name"],
                    test_case.get("sample_size", 1000),
                    output_file=target_file,
                    column_name=sample_by,
                )

                project = fc.create_project(project_name)
                upload = fc.upload_file(target_file, associate=project)
                project = wait_for_project(project)
                es = project.create_embedding_space(
                    name="ES test", 
                    wait_for_completion=True
                )

                assert es is not None
                # assert (
                #     job.finished is True
                # ), f"Job {job.id} did not finish with wait_for_completion"
                # assert job.error is False, f"Job {job.id} failed"

                uploads_to_delete.append(project)
                projects_to_delete.append(upload)
            except Exception as e:  # noqa
                print("########### ERROR #########################")
                print(f"Failed ES test case {test_idx} with error {e}")
                print(traceback.format_exc())
                print("########### ERROR #########################")
                if raise_on_error:
                    raise
    if verbose:
        print(
            f"\nTesting Embedding Space creation finished in {(datetime.utcnow()-start).total_seconds()}\n"
        )
    return


def cleanup():
    for p in projects_to_delete:
        p.delete()
    for u in uploads_to_delete:
        u.delete()


def ensure_featrixclient(ensure_pypi: bool):
    print(f"ensure_featrixclient: ensure_pypi = {ensure_pypi}")
    if ensure_pypi:
        try:
            from featrixclient.version import version, publish_time
        except ImportError:
            raise ImportError(
                "featrixclient not found in the Python path, trying to install from PyPI"
            )

        from featrixclient import __file__ as fc_file

        if "site-packages" in fc_file:
            from featrixclient.version import version, publish_time

            print(
                f"[1] Using featrixclient {version} installed in {Path(fc_file).parent}, built {publish_time}"
            )
            return
        raise ImportError(
            f"featrixclient is being loaded from {Path(fc_file).parent} not site-packages."
        )
    else:
        # try:
        #     import featrixclient
        # except ImportError:
        # We assume we are running the smoketest in the src tree under test
        srctop = str(top)
        print("srctop = ", srctop)
        sys.path.insert(0, srctop)

        try:
            import featrixclient
        except ImportError:
            path_str = '\n\t' + '\n\t'.join(sys.path)
            raise ImportError(
                f"Could not find featrixclient in the source tree, python path {path_str}"
            )
    print(
        f"[2] Using featrixclient from src in {Path(featrixclient.__file__).parent}"
    )
    return


def read_test_driver(args):
    import json

    if not args.data_file.exists():
        raise FileNotFoundError(f"Could not find test driver file {args.data_file}")
    lines = []
    for line in args.data_file.read_text().split("\n"):
        if line.strip().startswith("#"):
            continue
        r = line.rfind("#")
        if r > 0:
            line = line[:r]
        lines.append(line)
    tests = json.loads(args.data_file.read_text())
    if "uploads" not in tests:
        if args.verbose:
            print("No uploads test case found in {filename}, skipping upload testing")
        args.skip_uploads = True
    if "nf" not in tests:
        if args.verbose:
            print(
                "No neural function test case found in {filename}, skipping neural function testing"
            )
        args.skip_nf = True
    if "es" not in tests:
        if args.verbose:
            print(
                "No embedding space test case found in {filename}, skipping embedding space testing"
            )
        args.skip_es = True
    if "explorer" not in tests:
        if args.verbose:
            print(
                "No explorer test case found in {filename}, skipping explorer testing"
            )
        args.skip_explorer = True
    return args, tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featrix-host", help="dev (localhost) | Run against the specified Featrix host")
    ap.add_argument(
        "--no-clean", action="store_true", help="Do not clean up uploads and projects after running"
    )
    ap.add_argument(
        "--no-clean-on-error",
        "-E",
        action="store_true",
        help="Do not clean up the uploads and projects if there is an error",
    )
    ap.add_argument("--skip-uploads", action="store_true", help="Skip the upload tests")
    ap.add_argument(
        "--skip-nf", action="store_true", help="Skip the neural function tests"
    )
    ap.add_argument(
        "--skip-es", action="store_true", help="Skip the embedding space tests"
    )
    ap.add_argument(
        "--raise-on-error", "-X", action="store_true", help="Stop tests on any error"
    )
    ap.add_argument(
        "--data-dir",
        action="store",
        default=str(top / "tests/data"),
        help="Directory containing data files",
    )
    ap.add_argument(
        "--data-file",
        action="store",
        default=str(top / "tests/data/smoketest.json"),
        help="Data specific json data file to use for testing",
    )
    ap.add_argument(
        "--ensure-pypi",
        action="store_true",
        help="Ensure we are running against an installed version of featrixclient, not the local src",
        default=False
    )
    verbose = True
    # ap.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")
    args = ap.parse_args()

    ensure_featrixclient(args.ensure_pypi)
    args.data_dir = Path(args.data_dir)
    args.data_file = Path(args.data_file)
    args, test_cases = read_test_driver(args)
    fc = get_client(args.featrix_host)

    start = datetime.utcnow()
    try:
        if not args.skip_uploads:
            test_uploads(
                fc,
                args.data_dir,
                test_cases["uploads"],
                verbose=verbose,
                raise_on_error=args.raise_on_error,
            )
        if not args.skip_nf:
            test_nf(
                fc,
                args.data_dir,
                test_cases["nf"],
                raise_on_error=args.raise_on_error,
            )
        if not args.skip_es:
            test_es(
                fc,
                args.data_dir,
                test_cases["es"],
                verbose=verbose,
                raise_on_error=args.raise_on_error,
            )
        print(
            f"\nSmoke test complete in {(datetime.utcnow() - start).total_seconds()} seconds."
        )

    except Exception as e:
        import traceback

        print(f"Smoke test failed: {e}\n{traceback.format_exc()}")
        args.no_clean = args.no_clean_on_error
    if not args.no_clean:
        print("Cleaning up...")
        cleanup()


if __name__ == "__main__":
    main()
