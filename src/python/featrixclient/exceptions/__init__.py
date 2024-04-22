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
from typing import Any
from warnings import warn

from pydantic import ValidationError


class FeatrixException(Exception):
    pass


class FeatrixConfigException(FeatrixException):
    pass


class FeatrixNoSuchOrganization(FeatrixException):
    pass


class FeatrixNoSuchUser(FeatrixException):
    pass


class FeatrixPermissionDenied(FeatrixException):
    pass


class FeatrixDuplicateApiKeyLabel(FeatrixException):
    pass


class FeatrixNoGuestOrganization(FeatrixException):
    pass


class FeatrixDuplicateAssociation(FeatrixException):
    def __init__(self, project: Any, association: Any):
        self.association = association
        self.project = project
        super().__init__(
            f"Project {project.name} already has upload "
            f"association with {association.id}"
        )

    def details(self):
        return {
            "label": self.association.label,
            "project_name": self.project.name,
            "upload_id": self.association.upload_id,
            "sample_percentage": self.association.sample_percentage,
            "sample_row_count": self.association.sample_row_count,
            "drop_duplicates": self.association.drop_duplicates,
        }


class ProjectMappingsError(FeatrixException):
    def __init__(self, field_name, message):
        self.message = "%s: %s" % (field_name, message)


class DataSpaceMappingsError(FeatrixException):
    def __init__(self, field_name, message):
        self.message = "%s: %s" % (field_name, message)


class ProjectPreparationError(FeatrixException):
    pass


class NaNModelCollapseException(Exception):
    def __init__(self, colName):
        super().__init__(
            "Training model/vector space collapsed to NaNs on column %s" % (colName,)
        )
        self.colName = colName


class FeatrixConnectionError(Exception):
    def __init__(self, url, message):
        # logger.error("Connection error for url %s: __%s__" % (url, message))
        super().__init__("Connection error for URL %s: __%s__" % (url, message))


class FeatrixServerResponseParseError(Exception):
    def __init__(self, url, payload):
        # logger.error("Error parsing result from url %s: __%s__" % (url, payload))
        super().__init__("Bad response from URL %s: __%s__" % (url, payload))


class FeatrixBadServerCodeError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = "Unexpected status code %s: %s" % (status_code, message)
        super().__init__(self.message)


class FeatrixServerError(Exception):
    def __init__(self, msg):
        self.message = msg
        super().__init__(self.message)


class FeatrixNoApiKeyError(Exception):
    def __init__(self, msg):
        self.message = msg


class FeatrixBadApiKeyError(Exception):
    def __init__(self, msg):
        self.message = msg


class FeatrixEmbeddingSpaceNotSpecified(Exception):
    def __init__(self):
        self.message = "No embedding space id was specified or set. Call fit() or create an embedding space with an id first."
        super().__init__(self.message)


class FeatrixEmbeddingSpaceSpecified(Exception):
    def __init__(self):
        self.message = (
            "Calling fit() on an object with an existing id will change the id."
        )
        super().__init__(self.message)


class FeatrixModelNotSpecified(Exception):
    def __init__(self):
        self.message = (
            "No model id was specified. Call fit() or create a model with an id first."
        )
        super().__init__(self.message)


class FeatrixEmbeddingSpaceNotFound(FeatrixBadServerCodeError):
    """
    Embedding space not found on server.
    """

    def __init__(self, vector_space_id):
        self.message = f'Embedding space "{vector_space_id}" not found.'
        super().__init__(400, self.message)


class FeatrixDataSpaceNotFound(FeatrixBadServerCodeError):
    """
    Data space not found on server.
    """

    def __init__(self, data_space):
        self.message = f'Data space "{data_space}" not found.'
        super().__init__(400, self.message)


class FeatrixModelNotFound(FeatrixBadServerCodeError):
    """
    Model not found in embedding space.
    """

    def __init__(self, model_id, vector_space_id):
        self.message = (
            f'Model "{model_id}" not found in embedding space "{vector_space_id}".'
        )
        super().__init__(400, self.message)


class FeatrixDatabaseNotFound(FeatrixBadServerCodeError):
    """
    Specified database not found.
    """

    def __init__(self, db_id, vector_space_id):
        self.message = (
            f'Database "{db_id}" not found in embedding space "{vector_space_id}".'
        )
        super().__init__(400, self.message)


class FeatrixProjectNotFound(FeatrixBadServerCodeError):
    """
    Specified project not found.
    """

    def __init__(self, project_name: str):
        self.message = f'Project "{project_name}" not found.'
        super().__init__(400, self.message)


class FeatrixColumnNotFound(FeatrixBadServerCodeError):
    """
    Specified column not found.
    """

    def __init__(self, vector_space_id, col_name, all_col_names):
        self.message = f"Column \"{col_name}\" not found in embedding space \"{vector_space_id}\". not found. Available column names: {', '.join(all_col_names)}."
        super().__init__(400, self.message)


class FeatrixColumnNotAvailable(FeatrixBadServerCodeError):
    """
    Specified column not available. Typically this means we weren't able to encode or decode it.
    """

    def __init__(self, vector_space_id, col_name, encoded_columns):
        self.message = f"Column \"{col_name}\" not available for encoding in embedding space \"{vector_space_id}\". Available columns with codecs: {', '.join(encoded_columns)}."
        super().__init__(400, self.message)


class FeatrixInvalidModelQuery(FeatrixBadServerCodeError):
    def __init__(self, p):
        self.message = f"Invalid model query: {p}"
        super().__init__(400, self.message)


class FeatrixProjectExists(FeatrixBadServerCodeError):
    def __init__(self, p):
        self.message = f'Project "{p}" already exists.'
        super().__init__(400, self.message)


class FeatrixInvalidJob(FeatrixException):
    pass


def api_argument_error(exc: ValidationError, job_arg_cls=None):
    error_count = exc.error_count()
    errors = exc.errors()
    warn(
        f"{error_count} error{'s' if error_count > 1 else ''} occurred using API interface for {exc.title}:"
    )
    for error in errors:
        if error["type"] == "missing":
            warn(f"\t{error['msg']}: {error['loc']} is a required field")
        else:
            warn(f"\t{error['loc']}: {error['type']}: {error['msg']}")


def ParseFeatrixError(s):
    import json

    PREFIX = "featrix_exception:"
    if s.startswith(PREFIX):
        s = s[len(PREFIX) :]
        try:
            jr = json.loads(s)
        except:  # noqa E722
            return Exception(
                f"Internal error: Couldn't parse __{s}__ into Featrix exception"
            )
        error_name = jr.get("error_name")
        p = jr.get("params")
        if error_name == "embedding space not found":
            return FeatrixEmbeddingSpaceNotFound(p.get("vector_space_id"))
        elif error_name == "column not found":
            return FeatrixColumnNotFound(
                p.get("vector_space_id"), p.get("col_name"), p.get("all_col_names")
            )
        elif error_name == "column not encoded":
            return FeatrixColumnNotAvailable(
                p.get("vector_space_id"), p.get("col_name"), p.get("encoded_col_names")
            )
        elif error_name == "data space not found":
            return FeatrixDataSpaceNotFound(p.get("data_space_id"))
        elif error_name == "model not found in embedding space":
            return FeatrixModelNotFound(p.get("model_id"), p.get("vector_space_id"))
        elif error_name == "database not found in embedding space":
            return FeatrixDatabaseNotFound(p.get("db_id"), p.get("vector_space_id"))
        elif error_name == "invalid model query":
            return FeatrixInvalidModelQuery(p)
        elif error_name == "Project exists with the specified name already.":
            return FeatrixProjectExists(p.get("name"))
    # raise Exception(f"Unexpected error: {s}")
    return None
