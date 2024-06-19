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

        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
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
