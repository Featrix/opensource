#  -*- coding: utf-8 -*-
#############################################################################
#
#  Copyright (c) 2024, Featrix, Inc. All rights reserved.
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
#
#
#############################################################################
#
#  Yes, you can see this file, but Featrix, Inc. retains all rights.
#
#############################################################################
from __future__ import annotations

import csv
import os
import traceback
from io import StringIO
from pathlib import Path

import pandas as pd


def running_in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except:  # noqa - anyway this fails, we aren't in a notebook
        return False
    return True


def clear_cell(wait=False):
    try:
        from IPython.display import clear_output

        if running_in_notebook():
            clear_output(wait=wait)
    except ImportError:  # noqa
        pass


def display_message(msg: str):
    clear_cell(wait=False)
    print(msg)


def _find_bad_line_number(file_path: Path | str = None, buffer: bytes | str = None):
    try:
        if file_path:
            buffer = file_path.read_text()

        reader = csv.reader(buffer)
        line_number = 1
        try:
            for _ in reader:
                line_number += 1
        except Exception as e:  # noqa
            return line_number
    except:  # noqa nosec
        pass  # noqa: S110 nosec: B110
    return -1


# A wrapper for dealing with CSV files.
def log_trace(s):
    print("\tTRACE CSV:", s)


def count_newlines(s):
    parts = s.split("\n")
    return len(parts)


def check_excel_utf16_nonsense(file_path: str):
    with open(file_path, "rb") as fp:
        try:
            bytes = fp.read(16)
            if bytes[0] == 0xEF and bytes[1] == 0xBB and bytes[2] == 0xBF:
                # print("CRAZY STUFF MAN")
                return True
        except:  # noqa
            return False

    return False


def featrix_wrap_pd_read_csv(
    file_path: str | Path = None,
    buffer: bytes | str = None,
    on_bad_lines="skip",
    trace=False,
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
    trace: bool
        Trace path through this code for debugging on problematic files.

    This can raise exceptions if the file is not found or seems to be empty.

    """
    df = None
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

        hasSillyStuff = check_excel_utf16_nonsense(file_path)
        if hasSillyStuff:
            new_file = file_path + ".cleaned"
            with open(new_file, "wb") as fp_new:
                with open(file_path, "rb") as fp_old:
                    fp_old.seek(3)
                    try:
                        data = fp_old.read()
                        fp_new.write(data)
                    except:  # noqa
                        print("error copying file...")
                        traceback.print_exc()
            if trace:
                log_trace(
                    f"new file created without Excel bytes: {file_path} -> {new_file}"
                )
            file_path = new_file
    elif isinstance(buffer, bytes):
        buffer = buffer.decode()

    buffer_io = StringIO(buffer) if buffer else None
    dialect = None
    has_header = True

    input_size = None
    if buffer:
        buffer_io.seek(0, os.SEEK_END)
        input_size = buffer_io.tell()
        buffer_io.seek(0)
    else:
        input_size = os.path.getsize(file_path)
        if trace:
            log_trace(f"input_size of {file_path} --> {input_size}")

    # get the file size and adjust the sampling based on that.
    assert input_size is not None  # nosec

    sample_size = 32 * 1024
    if input_size < sample_size:
        sample_size = input_size - 1
    # else:
    #     if input_size > (1024 * 1024):
    #         sample_size = 256 * 1024

    if trace:
        log_trace(f"sample_size = {sample_size}")

    # sniff
    sniffer = csv.Sniffer()
    if buffer:
        if trace:
            log_trace("buffer != None")

        attempts = 0
        while attempts < 10:
            if trace:
                log_trace(f"buffer read: attempt {attempts}")
            attempts += 1
            try:
                buffer_io.seek(0)
                sample_buffer = buffer_io.read(sample_size)
                num_newlines = count_newlines(sample_buffer)
                if num_newlines < 10:
                    if trace:
                        log_trace(
                            f"attempt={attempts}: Only {num_newlines} found in first {sample_size} bytes--increasing buffer size"
                        )
                    sample_size *= 2
                    if sample_size > input_size:
                        if trace:
                            log_trace("hitting the end of the buffer...")
                        sample_size = input_size - 1
                    continue
                else:
                    # OK, we have at least 4 lines. we're good
                    if attempts > 1:
                        log_trace(
                            f"attempt={attempts}: Found {num_newlines} in first {sample_size} bytes--proceeding"
                        )

                    dialect = sniffer.sniff(sample_buffer)
                    has_header = sniffer.has_header(sample_buffer)
                    break
            except:  # noqa
                traceback.print_exc()
                print("Not sure what to do here.")
                pass  # nosec
        if trace:
            log_trace(
                f'buffer sniffer: header = {has_header}, dialect delimiter = "{dialect.delimiter}"'
            )
    else:
        # check for a gzip header first.
        def has_gzip_header(buffer):
            # GZIP files start with these two bytes
            GZIP_MAGIC_NUMBER = b"\x1f\x8b"
            return buffer.startswith(GZIP_MAGIC_NUMBER)

        if trace:
            log_trace("checking if it's a gzip file...")
        with open(file_path, "rb") as gzip_file:
            possible_header = gzip_file.read(10)
            if has_gzip_header(possible_header):
                if trace:
                    log_trace("we got a gzip header, trying to gunzip it...")

                # uncompress it first.
                # print(f"file_path = __{file_path}__")
                # print(os.system(f"ls {str(file_path)}*"))
                os.rename(file_path, f"{file_path}.gz")
                rc = os.system(f'gunzip -k -t "{file_path}.gz"')  # nosec -- yikes
                log_trace(f"gunzip returned {rc}")
                file_path = Path(str(file_path) + ".gz")

        log_trace(f"working with file_path = {file_path}")
        with open(file_path, newline="", errors="ignore") as csvfile:
            attempts = 0
            all_good = False
            while attempts < 10:
                if all_good:
                    if trace:
                        log_trace("all_good is True")
                    break
                if trace:
                    log_trace(f"file read: attempt {attempts} to read {file_path}")

                if sample_size > input_size:
                    if trace:
                        log_trace("hitting the end of the buffer...")
                    sample_size = input_size - 1

                attempts += 1
                try:
                    csvfile.seek(0)
                    sample = csvfile.read(sample_size)
                    num_newlines = count_newlines(sample)
                    if trace:
                        log_trace(f"got num_newlines = {num_newlines}")

                    if num_newlines < 10:
                        if trace:
                            log_trace(
                                f"attempt={attempts}: Only {num_newlines} found in first {sample_size} bytes--increasing buffer size"
                            )
                        sample_size *= 2
                        continue
                    else:
                        # OK, we have at least 4 lines. we're good
                        if attempts > 1:
                            if trace:
                                log_trace(
                                    f"attempt={attempts}: Found {num_newlines} in first {sample_size} bytes--proceeding"
                                )
                except:  # noqa
                    bad_line = _find_bad_line_number(file_path=file_path, buffer=buffer)
                    if bad_line > 0:
                        if trace:
                            log_trace("first BAD LINE WAS ...{bad_line}")

                try:
                    dialect = sniffer.sniff(sample)
                    has_header = sniffer.has_header(sample)
                    all_good = True
                except Exception as err:
                    sample_size *= 2
                    if trace:
                        log_trace(
                            f"attempt={attempts}: got an error: {err}... will try again... new sample size is {sample_size}"
                        )
    if trace:
        log_trace(
            f"file sniffer: sample length = {len(sample)}, header = {has_header}, dialect delimiter = \"{dialect.delimiter if dialect is not None else 'None'}\""
        )

    csv_parameters = {}
    if dialect is not None:
        csv_parameters = {
            "delimiter": dialect.delimiter,
            "quotechar": dialect.quotechar,
            "escapechar": dialect.escapechar,
            "doublequote": dialect.doublequote,
            "skipinitialspace": dialect.skipinitialspace,
            "quoting": dialect.quoting,
            # Pandas does not support line terminators > 1 but Sniffer returns things like '\r\n'
            # 'lineterminator': dialect.lineterminator
        }

    # print("has_header = ", has_header)
    # print("csv_parameters = ", csv_parameters)

    if has_header:
        if trace:
            log_trace(f"has_header = {has_header}")
        try:
            df = pd.read_csv(
                file_path or buffer_io,
                # Pandas doesn't take the same dialect as csv.Sniffer produces so we create csv_parameters
                # dialect=dialect,
                on_bad_lines=on_bad_lines,
                encoding_errors="ignore",
                **csv_parameters,
            )
            if trace:
                log_trace(f"{file_path}: loaded {len(df)} x {len(df.columns)}")

            cols = list(df.columns)
            if len(cols) <= 1:
                if trace:
                    log_trace(
                        f"only got {len(cols)} ... trying without our sniffed parameters"
                    )
                # ok try it without the parameters.
                df = pd.read_csv(
                    file_path or buffer_io,
                    # Pandas doesn't take the same dialect as csv.Sniffer produces so we create csv_parameters
                    # dialect=dialect,
                    on_bad_lines=on_bad_lines,
                    encoding_errors="ignore",
                )

                if trace:
                    log_trace(f"{file_path}: loaded {len(df)} x {len(df.columns)}")
        except pd.errors.ParserError as err:
            if trace:
                log_trace(f"{file_path} - got a pandas parser error: {err}")

            try:
                df = pd.read_csv(
                    file_path or buffer_io,
                    # Pandas doesn't take the same dialect as csv.Sniffer produces so we create csv_parameters
                    # dialect=dialect,
                    on_bad_lines=on_bad_lines,
                    encoding_errors="ignore",
                )
                if trace:
                    log_trace(f"{file_path}: loaded {len(df)} x {len(df.columns)}")
            except:  # noqa
                traceback.print_exc()
                if trace:
                    log_trace("tried again with no parameters and still had an error")

        except csv.Error as err:
            bad_line = _find_bad_line_number(file_path=file_path, buffer=buffer)
            if bad_line > 0:
                if trace:
                    log_trace(f"read_csv() - first BAD LINE WAS ...{bad_line}")

            s_err = str(err)
            if trace:
                log_trace(f"read_csv() -> error -> {s_err}")

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
                    **csv_parameters,
                )
                if trace:
                    log_trace(f"{file_path}: loaded {len(df)} x {len(df.columns)}")
            else:
                if trace:
                    log_trace("c'est la vie")
                raise err
            # endif

        # if any of the columns have an 'int' type, rename it.
        if df is not None:
            if trace:
                log_trace("df is not None... doing rename of crazy columns")
            cols = list(df.columns)
            renames = {}
            for idx, c in enumerate(cols):
                if not isinstance(c, str):
                    renames[c] = "column_" + str(c)
            if len(renames) > 0:
                if trace:
                    log_trace(f"renaming some columns: {renames}")
                df.rename(columns=renames, inplace=True)

        if trace:
            log_trace(f"returning {df}")
        return df

    if not has_header:
        if trace:
            log_trace(f"has_header = {has_header}")
            log_trace(f"trying again -- csv_parameters = {csv_parameters}")

        try:
            df = pd.read_csv(file_path or buffer_io, **csv_parameters)
            if trace:
                log_trace(f"{file_path}: loaded {len(df)} x {len(df.columns)}")

            cols = list(df.columns)
            if len(cols) >= 0:
                count_unnamed_columns = 0
                for col_name in cols:
                    if col_name.startswith("Unnamed"):
                        count_unnamed_columns += 1

                if count_unnamed_columns == len(cols):
                    # still no good.
                    raise Exception(
                        f"CSV file {file_path} doesn't seem to have a header line, which means it does not "
                        "have labels for the columns. This will make creating predictions on "
                        "specific targets difficult!"
                    )
            return df
        except Exception as err:  # noqa - catch anything
            if trace:
                log_trace("trying again with no parameters specified")

            # OK, maybe the csv parameters are crap.
            df = pd.read_csv(file_path or buffer_io)
            if trace:
                log_trace(f"{file_path}: loaded {len(df)} x {len(df.columns)}")
            cols = list(df.columns)

            if trace:
                log_trace(f"df = {len(df)} rows x = {cols} cols")
            if len(cols) >= 0:
                count_unnamed_columns = 0
                for col_name in cols:
                    if col_name.startswith("Unnamed"):
                        count_unnamed_columns += 1

                if count_unnamed_columns == len(cols):
                    raise Exception(
                        f"CSV file {file_path} doesn't seem to have a header line, which means it does not "
                        "have labels for the columns. This will make creating predictions on specific targets difficult! [2]"
                    )
                return df
            else:
                raise Exception(
                    f"CSV file {file_path} doesn't seem to have a multiple columns that we could detect"
                )

    if trace:
        log_trace(f"returning {df} at the bitter end")
    return df
