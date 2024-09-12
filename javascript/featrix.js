///  -*- coding: utf-8 -*-
/// #############################################################################
///
///  Copyright (c) 2024, Featrix, Inc.
///
///  Permission is hereby granted, free of charge, to any person obtaining a copy
///  of this software and associated documentation files (the "Software"), to deal
///  in the Software without restriction, including without limitation the rights
///  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
///  copies of the Software, and to permit persons to whom the Software is
///  furnished to do so, subject to the following conditions:
///
///  The above copyright notice and this permission notice shall be included in all
///  copies or substantial portions of the Software.
///
///  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
///  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
///  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
///  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
///  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
///  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
///  SOFTWARE.
/// #############################################################################
///
///  Welcome to...
///
///   _______ _______ _______ _______ ______ _______ ___ ___
///  |    ___|    ___|   _   |_     _|   __ \_     _|   |   |
///  |    ___|    ___|       | |   | |      <_|   |_|-     -|
///  |___|   |_______|___|___| |___| |___|__|_______|___|___|
///
///                                                Let's embed!
///
/// #############################################################################
///
///  Sign up for Featrix at https://app.featrix.com/
///
/// #############################################################################
///
///  Check out the docs -- you can either call the built-in help() method
///  or fire up your browser:
///
///     https://featrix-docs.readthedocs.io/en/latest/
///
///  You can also join our community Slack:
///
///     https://bits.featrix.com/slack
///
///  We'd love to hear from you: bugs, features, questions -- send them along!
///
///     hello@featrix.ai
///
/// #############################################################################

class FeatrixConnectionError extends Error {
  constructor(url, message) {
    super(`Connection error for URL ${url}: ${message}`);
    this.name = "FeatrixConnectionError";
  }
}

class FeatrixBadApiKeyError extends Error {
  constructor(message) {
    super(message);
    this.name = "FeatrixBadApiKeyError";
  }
}

class Featrix {
  constructor({
    url = "https://app.featrix.com",
    clientId = null,
    clientSecret = null,
    allowUnencryptedHttp = false,
    debug = false,
  } = {}) {
    this.url = this._validateUrl(url, allowUnencryptedHttp);
    this.clientId = clientId;
    this.clientSecret = clientSecret;
    this.debug = debug;
    this.hostname = window.location.hostname;
    this.currentBearerToken = null;
    this.currentBearerTokenExpiration = null;

    return this._generateBearerToken().then(() => {
      if (!this.currentBearerToken) {
        throw new Error("Your client id and client secret pair are invalid.");
      }
    });
  }

  // Generate Bearer Token
  async _generateBearerToken() {
    if (this.debug) {
      console.log("_generate_bearer_token entered");
    }

    const headers = this._featrixHeaders(true);
    const payload = JSON.stringify({
      client_id: this.clientId,
      client_secret: this.clientSecret,
    });

    const response = await fetch(`${this.url}/mosaic/keyauth/jwt`, {
      method: "POST",
      headers: headers,
      body: payload,
    });

    if (response.ok) {
      const body = await response.json();
      this.currentBearerToken = body.jwt;
      this.currentBearerTokenExpiration = new Date(body.expiration);

      if (this.debug) {
        console.log("bearer looks good");
      }
    } else {
      throw new FeatrixBadApiKeyError(
        `Failed to create authorization token from API Key: ${response.status}. Are your client id and client secret correct?`,
      );
    }
  }

  // Generate Featrix Headers
  _featrixHeaders(bearerGenerate = false, jsonRequest = false, extra = {}) {
    let headers = {};

    if (jsonRequest) {
      headers = {
        "Content-Type": "application/json",
        Accept: "text/plain",
        "X-request-id": this._generateUUID(),
        "X-hostname": this.hostname,
      };
    }

    if (!bearerGenerate) {
      if (
        !this.currentBearerToken ||
        (this.currentBearerTokenExpiration &&
          this.currentBearerTokenExpiration < new Date())
      ) {
        return this._generateBearerToken();
      }

      if (!this.currentBearerToken) {
        throw new FeatrixBadApiKeyError(
          "Your API Key seems to have been invalidated, please create another one.",
        );
      }

      headers["Authorization"] = `Bearer ${this.currentBearerToken}`;
    }

    return Object.assign(headers, extra);
  }

  // UUID Generator (replaces uuid.uuid1())
  _generateUUID() {
    return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, (c) =>
      (
        c ^
        (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))
      ).toString(16),
    );
  }

  // Validate URL
  _validateUrl(url, allowUnencryptedHttp) {
    if (!url) {
      throw new Error("url argument must be in the form of https://host:port");
    }

    url = url.replace(/\/+$/, ""); // Remove trailing slashes
    this._validateUrlScheme(url, allowUnencryptedHttp);

    return `${url}/api`;
  }

  // Validate URL Scheme (HTTPS vs HTTP)
  _validateUrlScheme(url, allowUnencryptedHttp) {
    if (!/^https?:\/\//.test(url)) {
      throw new Error("url argument must be in the form of https://host:port");
    } else if (url.startsWith("http://")) {
      this._validateLocalhostUrl(url, allowUnencryptedHttp);
    }
  }

  // Validate Localhost URLs for HTTP
  _validateLocalhostUrl(url, allowUnencryptedHttp) {
    if (!allowUnencryptedHttp) {
      const localhostPrefixes = ["localhost", "127.0.0.1", "::1"];
      if (
        !localhostPrefixes.some(
          (lh) =>
            url === `http://${lh}` ||
            url.startsWith(`http://${lh}/`) ||
            url.startsWith(`http://${lh}:`),
        )
      ) {
        throw new Error(
          "Non-HTTPS only supported for localhost without setting `allow_unencrypted_http`.",
        );
      }
    }
  }

  // Prediction function
  async predict(neuralFunctionId, query) {
    if (typeof query === "object" && !Array.isArray(query)) {
      query = [query];
    }

    const headers = this._featrixHeaders();
    const url = `${this.url}/neural/models/prediction`;

    if (this.debug) {
      console.log("HTTP: ", url);
      console.log("headers: ", headers);
    }

    const payload = JSON.stringify({
      job_type: "model-prediction",
      model_id: neuralFunctionId,
      query: query,
    });

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: headers,
        body: payload,
      });

      if (this.debug) {
        console.log(`Response status: ${response.status}`);
      }

      if (!response.ok) {
        throw new FeatrixConnectionError(
          url,
          `Unexpected HTTP status: ${response.status}`,
        );
      }

      return await response.json();
    } catch (error) {
      throw new FeatrixConnectionError(url, error.message);
    }
  }
}
