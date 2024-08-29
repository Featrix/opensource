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
#     https://join.slack.com/t/featrixcommunity/shared_invite/zt-28b8x6e6o-OVh23Wc_LiCHQgdVeitoZg
#
#  We'd love to hear from you: bugs, features, questions -- send them along!
#
#     hello@featrix.ai
#
#############################################################################
#
from __future__ import annotations

import base64
import html
import json
import logging
import os
import socket
import uuid
import warnings
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import requests
from pydantic import BaseModel

from .api_urls import ApiInfo
from .config import settings
from .exceptions import FeatrixBadApiKeyError
from .exceptions import FeatrixConnectionError
from .exceptions import FeatrixException
from .exceptions import FeatrixNoApiKeyError
from .models import ChainedJobType
from .models import CreateDBArgs
from .models import DBClusterArgs
from .models import EncodeRecordsArgs
from .models import ESCreateArgs
from .models import GuardRailsArgs
from .models import JobType
from .models import ModelCreateArgs
from .models import ModelPredictionArgs
from .models import NewNeuralFunctionArgs
from .models import NNQueryArgs
from .models import TrainMoreArgs

logger = logging.getLogger(__name__)


class FeatrixApi:
    current_instance = None
    debug: bool = False

    job_args: Dict = {
        JobType.JOB_TYPE_ES_CREATE: (ESCreateArgs, "neural/embedding_space/"),
        JobType.JOB_TYPE_ES_TRAIN_MORE: (
            TrainMoreArgs,
            "neural/embedding_space/trainmore",
        ),
        JobType.JOB_TYPE_MODEL_CREATE: (
            ModelCreateArgs,
            "neural/embedding_space/train-downstream-model",
        ),
        JobType.JOB_TYPE_MODEL_PREDICTION: (
            ModelPredictionArgs,
            "neural/embedding_space/run-model-prediction",
        ),
        JobType.JOB_TYPE_MODEL_GUARDRAILS: (
            GuardRailsArgs,
            "neural/embedding_space/check-guardrails",
        ),
        JobType.JOB_TYPE_ENCODE_RECORDS: (
            EncodeRecordsArgs,
            "neural/embedding_space/run-encode-records",
        ),
        JobType.JOB_TYPE_ES_CREATE_DB: (
            CreateDBArgs,
            "neural/embedding_space/create-database",
        ),
        JobType.JOB_TYPE_DB_CLUSTER: (
            DBClusterArgs,
            "neural/embedding_space/run-db-cluster",
        ),
        JobType.JOB_TYPE_DB_NN_QUERY: (NNQueryArgs, "neural/embedding_space/nn-query"),
        ChainedJobType.CHAINED_JOB_TYPE_NNF: (
            NewNeuralFunctionArgs,
            "neural/project/new-neural-function",
        ),
    }

    job_urls: Dict = {}

    def __init__(
        self,
        featrix_client,
        url="https://app.featrix.com",
        client_id=None,
        client_secret=None,
        key_file: Optional[Path | str] = None,
        allow_unencrypted_http=False,
        debug: bool = False,
    ):
        self.debug = debug
        self.client = featrix_client
        self.client_id = client_id or settings.client_id
        self.client_secret = client_secret or settings.client_secret

        self.local_only = False
        self._queue_errors = []
        self._cached_hostname = None
        self.key_file = Path(key_file).expanduser() if key_file else None

        self._current_bearer_token = None
        self._current_bearer_token_expiration = None
        self.hostname = socket.gethostname()
        self.url = self._validate_url(url, allow_unencrypted_http)
        # Initialize authentication
        self._api_key_init(
            client_id, client_secret, Path(key_file) if key_file else None
        )
        if self.debug:
            print(f"Using base url {self.url}")
        backend = self.op("info_get")
        self.api_version = backend["version"]
        if self.debug:
            print(f"Connected to backend, version {self.api_version}")
        self.user = self.op("users_get_self")
        self.organization = self.op("org_get", org_id=self.user.current_organization_id)

    @classmethod
    def new(
        cls,
        featrix_client,
        url: str = "https://app.featrix.com",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        key_file: Optional[Path] = None,
        allow_unencrypted_http: bool = False,
        debug: bool = False,
    ):
        if cls.current_instance:
            # FIXME: we want to resolve the client_id and client_secret - if they changed, they want a new
            # FIXME: instance with different creds.
            client_id, client_secret = cls.ensure_key_file(
                client_id, client_secret, key_file
            )
            if (
                client_id == cls.current_instance.client_id
                and client_secret == cls.current_instance.client_secret
            ):
                return cls.current_instance
        cls.current_instance = cls(
            featrix_client,
            url=url,
            client_id=client_id,
            client_secret=client_secret,
            key_file=key_file,
            allow_unencrypted_http=allow_unencrypted_http,
        )
        return cls.current_instance

    @property
    def instance(self):
        return self.current_instance

    def op(self, api_call, *args, **kwargs) -> Any:
        # print(f"{api_call} -- {args}  kw {kwargs}")
        arguments = files = None
        api = ApiInfo().get(api_call)
        # get, post, delete, etc
        verb = ApiInfo.verb(api_call)
        # do we need to have a separate submit_job?
        if verb == "job":
            # FIXME: do we want to separate jobs?  For now we just push them as posts, but we keep this
            # so we can do something special with jobs if we want.
            verb = "post"
            # return self.submit_job(job_args=api.arg_type(**kwargs))
        if api is None:
            raise RuntimeError(f"No such API call {api_call}")
        # do url call - the get/post/create/delete is in the api_call name, and the api above has
        # ['url', 'arg_type', 'response_type'] -- probably some extra work for a few around "arg_type" but for the
        # most part, we should  be able to stand up arg_type from kwargs, and convert the result to "response_type",
        if api.arg_type is None:
            pass
        elif api.arg_type == "files":
            files = kwargs
        elif isinstance(api.arg_type, dict):
            arguments = kwargs
        elif len(args) > 0 and isinstance(args[0], api.arg_type):
            arguments = args[0]
        else:
            arguments = api.arg_type(**kwargs)

        # print(f"Calling _op with args type {type(arguments)}")
        url = f"{self.url}{ApiInfo.url_substitution(api.url, **kwargs)}"
        response_data = self._op(verb, url, self._featrix_headers(), arguments, files)
        return ApiInfo.featrix_validate(api_call, response_data)

    @staticmethod
    def path_options(url: str, args: Dict) -> Tuple[str, Dict | None]:
        if args is None:
            return url, None
        d = args.pop("since", None)
        if d is not None:
            if isinstance(d, datetime):
                url += f"?since={d.isoformat()}"
            else:
                url += f"?since={d}"
        # FIXME: count, sort, etc
        return url, args

    def _op(
        self,
        verb: str,
        url: str,
        headers: Dict,
        args: Optional[Dict],
        files: Optional[Dict],
        retries: int = 3,
    ):
        try:
            # print(f"OP: {verb}: {url} args {args}")
            if isinstance(args, BaseModel):
                args = args.model_dump()
            url, args = self.path_options(url, args)
            if self.debug:
                print(
                    f"Issuing request {verb}:{url} -- json={args} files={'yes' if files else None} "
                    f"headers={list(headers.keys())}"
                )
            response = requests.request(
                verb, url, headers=headers, json=args, files=files
            )
            if self.debug:
                print(f"Response status: {response.status_code}")
            # print(f"response back {response}")
        except requests.exceptions.Timeout:
            if self.debug:
                print("Response exception: Timeout")
            retries -= 1
            warnings.warn(f"Request timed out, retrying (will retry {retries} times")
            response = None
        except requests.exceptions.HTTPError as err:
            if self.debug:
                print(f"Response exception: {err}")
            raise FeatrixConnectionError(url, f"http error: {err}")
        except requests.RequestException as e:
            if self.debug:
                print(f"Response exception: {e}")
            raise FeatrixConnectionError(url, f"request error: {e}")
        except Exception as e:
            if self.debug:
                print(f"Response exception: {e}")
            raise FeatrixConnectionError(url, f"Unknown error {e}")

        if response is not None:
            if self.debug:
                print(f"Processing response with status: {response.status_code}")
            if response.status_code in (HTTPStatus.OK, HTTPStatus.CREATED):
                return self.fix_ids(response.json())
            elif response.status_code == HTTPStatus.UNAUTHORIZED:
                self._generate_bearer_token()
                return self._op(verb, url, headers, args, files, retries - 1)
            elif response.status_code == HTTPStatus.BAD_REQUEST:
                err_text = self._parse_html_crazy(response.text)  # ??
                raise FeatrixException(f"Bad request: {err_text}")
            else:
                err_text = self.error_message(response) or str(response.status_code)
                raise FeatrixException(f"Error with request: {err_text}")
            # special_exception = ParseFeatrixError(err_text)
            # if special_exception is not None:
            #     raise special_exception
        if retries > 0:
            return self._op(verb, url, headers, args, files, retries=retries - 1)
        raise FeatrixConnectionError(
            url,
            f"No more retries, multiple errors {response.status_code if response else ''}",
        )

    def fix_ids(self, data: Any):
        if isinstance(data, list):
            return [self.fix_ids(_) for _ in data]
        if isinstance(data, dict):
            if "_id" in data:
                data["id"] = data["_id"]
        return data

    def log_activity(self):
        pass

    def _api_key_init(self, client_id: str, client_secret: str, key_file: Path):
        self.client_id, self.client_secret = self.ensure_key_file(
            client_id, client_secret, key_file
        )
        if self._current_bearer_token:
            return
        self._generate_bearer_token()
        if self._current_bearer_token is None:
            raise FeatrixBadApiKeyError(
                "Your client id and client secret pair are invalid."
            )
        return

    def _generate_bearer_token(self):
        headers = self._featrix_headers(bearer_generate=True)
        response = requests.post(
            f"{self.url}/mosaic/keyauth/jwt",
            headers=headers,
            data=json.dumps(
                {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                }
            ),
        )

        if response.status_code == 200:
            # The response will have your JWT in it -- you can use that now for 24 hours
            # If this was a guest access, it will also have the client-id/client-secret for reuse
            body = response.json()
            self._current_bearer_token = body["jwt"]
            self._current_bearer_token_expiration = datetime.fromisoformat(
                body["expiration"]
            )
            if self.client_id == "guest":
                self.client_id = body[0].get("client_id")
                self.client_secret = body[0].get("client_secret")
                self._save_key_file()
        else:
            raise FeatrixBadApiKeyError(
                f"Failed to create authorization token from API Key: {response.status_code}. "
                "Are your client id and client secret correct?"
            )

    def _check_bearer_token(self, header):
        """
        In the case that the server returns an Authorization header, it means the server refreshed the token
        for us, so update it and save ourselves having to deal with it in the next GET/POST.
        """
        if header is None:
            return
        if header.startswith("Bearer"):
            token = header.replace("Bearer ", "").strip()
            self._current_bearer_token = token
        return

    @staticmethod
    def extract_keys_from_environment(
        client_id: str | None, client_secret: str | None
    ) -> Tuple[str, str]:
        if not client_id:
            client_id = os.environ.get("FEATRIX_CLIENT_ID")
        if not client_secret:
            client_secret = os.environ.get("FEATRIX_CLIENT_SECRET")
        return client_id, client_secret

    @staticmethod
    def parse_key_file(key_file: Path) -> Tuple[str | None, str | None]:
        client_id = None
        client_secret = None
        if key_file is None or not key_file.exists():
            return None, None
        for line in key_file.read_text().splitlines():
            tag, _, key_value = line.partition("=")
            if len(key_value):
                if "FEATRIX_CLIENT_ID" in tag.upper():
                    client_id = key_value.strip()
                elif "FEATRIX_CLIENT_SECRET" in tag.upper():
                    client_secret = key_value.strip()
        return client_id, client_secret

    @staticmethod
    def ensure_key_file(
        client_id: str, client_secret: str, key_file: Path | None
    ) -> Tuple[str, str]:
        """
        If the user did not pass in a client id/secret, we look in the environment.  If it is missing from
        the environment, we look in the key file.  The precedence is init params, environment, file.  We
        do only use a pair in both cases so if the user passes in the id, but we find a pair in the file,
        we use both values from the file.

        If no values are provided, we try to use guest access but if only one value shows up, we error out.

        Arguments:
            client_id(str): the client id portion of your api key
            client_secret(str): the client secret portion of your api key
            key_file: optional Path() to key file
        """
        client_id, client_secret = FeatrixApi.extract_keys_from_environment(
            client_id, client_secret
        )
        if None in [client_id, client_secret]:
            client_id, client_secret = FeatrixApi.parse_key_file(key_file)

        # if client_id is None and client_secret is None:
        #     self._register_guest()

        if None in [client_id, client_secret]:
            raise FeatrixNoApiKeyError(
                "You provided only part of an API Key pair. If you want to use the guest capabilities "
                "make sure you don't have FEATRIX_CLIENT_ID or FEATRIX_CLIENT_SECRET in your environment "
                "and that you don't have a ~/.featrix.key file in place"
            )
        settings.client_id = client_id
        settings.client_secret = client_secret
        return client_id, client_secret

    def _save_key_file(self):
        """
        Save the guest key to a file, so we can reuse it if the key file doesn't exist.
        """
        if None in [self.client_id, self.client_secret] or "guest" in [
            self.client_id,
            self.client_secret,
        ]:
            return
        if self.key_file is None:
            return
        if not self.key_file.exists():
            with self.key_file.open("w") as _file:
                _file.write(
                    f"FEATRIX_CLIENT_ID={self.client_id}\n"
                    f"FEATRIX_CLIENT_SECRET={self.client_secret}\n"
                )
        return

    def _featrix_headers(
        self,
        extra: Optional[Dict] = None,
        json_request: Optional[bool] = False,
        file_request: Optional[str] = None,
        bearer_generate: Optional[bool] = False,
        **kwargs,
    ):
        # FIXME: do we use this anymore?
        if file_request:
            headers = {
                "Content-type": "text/csv",
                "Accept": "text/plain",
                "X-agent": "featrix-client",
                "X-data-type": "csv",
                "X-batch-name": base64.urlsafe_b64encode(
                    file_request.encode("utf-8")
                ).decode("utf-8"),
            }
        elif json_request:
            headers = {
                "Content-type": "application/json",
                "Accept": "text/plain",
                "X-request-id": str(uuid.uuid1()),
                "X-hostname": self.hostname,
            }
        else:
            headers = {}

        if extra is not None:
            headers.update(extra)
        headers.update(kwargs)
        if not bearer_generate:
            if self._current_bearer_token is None or (
                self._current_bearer_token_expiration
                and self._current_bearer_token_expiration < datetime.now()
            ):
                self._generate_bearer_token()
            if self._current_bearer_token is None:
                raise FeatrixBadApiKeyError(
                    "You ApiKey seems to have been invalidated, please create another one"
                )
            # Don't let old/handed-in headers overwrite the auth token
            headers["Authorization"] = f"Bearer {self._current_bearer_token}"
        return headers

    @staticmethod
    def _retry_on_error_code(rc):
        return 400 <= rc < 500

    @staticmethod
    def _parse_html_crazy(new_text):
        # Chop off the annoying HTML nonsense if it's there..
        if new_text.startswith("<!DOCTYPE") or new_text.startswith("<!doctype"):
            p_pos = new_text.find("<p>")
            if p_pos >= 0:
                new_text = new_text[p_pos + 3 :]
                p_pos = new_text.find("</p>")
                if p_pos > 0:
                    new_text = new_text[:p_pos]
                    new_text = html.unescape(new_text)
                    return new_text
        return "Could not parse: " + str(new_text)

    @staticmethod
    def error_message(response):
        try:
            msg = response.json()
            if isinstance(msg, dict):
                txt = msg.get('detail', msg.get('message', msg.get('error', msg)))
            else:
                txt = msg
            return txt
        except Exception:
            return response.text

    def __del__(self):
        logger.debug("Featrix destructor called")

    def _validate_url(self, url: str, allow_unencrypted_http) -> str:
        if url is None:
            raise RuntimeError(
                f"url argument to {self.__class__.__name__} must be in "
                "the form of https://host:port"
            )
        url = url.rstrip("/")  # Remove trailing slashes
        self._validate_url_scheme(url, allow_unencrypted_http)
        return f"{url}/api"

    def _validate_url_scheme(self, url: str, allow_unencrypted_http):
        if not url.startswith("http://") and not url.startswith("https://"):  # noqa
            raise RuntimeError(
                "url argument to must be in the form of https://host:port"
            )
        elif url.startswith("http://"):  # noqa - dev mode
            self._validate_localhost_url(url, allow_unencrypted_http)
        return

    @staticmethod
    def _validate_localhost_url(url: str, allow_unencrypted_http):
        if not allow_unencrypted_http:
            localhost_prefixes = ["localhost", "127.0.0.1", "::1"]
            for lh in localhost_prefixes:
                lh = "http://" + lh  # noqa dev mode
                # Capture with both a specified port or a slash.
                if url == lh or url.startswith(lh + "/") or url.startswith(lh + ":"):
                    break
            else:
                raise RuntimeError(
                    "Non-HTTPS only supported for localhost without setting `allow_unencrypted_http`"
                )
        return
