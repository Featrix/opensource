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

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Runtime.CompilerServices;

namespace FeatrixExample
{
    public class FeatrixConnectionError : Exception
    {
        public FeatrixConnectionError(string url, string message)
            : base($"Connection error for URL {url}: __{message}__") { }
    }

    public class FeatrixBadApiKeyError : Exception
    {
        public FeatrixBadApiKeyError(string message) : base(message) { }
    }

    public class Featrix
    {
        private string _url;
        private string _clientId;
        private string _clientSecret;
        private string _hostname;
        private string _currentBearerToken;
        private DateTime? _currentBearerTokenExpiration;
        private bool _debug;

        private static readonly HttpClient httpClient = new HttpClient();

        public Featrix(
            string url = "https://app.featrix.com",
            string clientId = null,
            string clientSecret = null,
            bool allowUnencryptedHttp = false,
            bool debug = false
        )
        {
            _debug = debug;
            _clientId = clientId;
            _clientSecret = clientSecret;
            _url = ValidateUrl(url, allowUnencryptedHttp);
            _hostname = Dns.GetHostName();
            GenerateBearerTokenAsync().Wait();

            if (_currentBearerToken == null)
            {
                throw new Exception("Your client id and client secret pair are invalid.");
            }
        }

        private async Task GenerateBearerTokenAsync()
        {
            if (_debug)
            {
                Console.WriteLine("_generate_bearer_token entered");
            }

            var headers = FeatrixHeaders(bearerGenerate: true);

            var payload = JsonSerializer.Serialize(new
            {
                client_id = _clientId,
                client_secret = _clientSecret
            });

            var response = await httpClient.PostAsync($"{_url}/mosaic/keyauth/jwt", new StringContent(payload, Encoding.UTF8, "application/json"));

            if (response.IsSuccessStatusCode)
            {
                var body = JsonSerializer.Deserialize<Dictionary<string, string>>(await response.Content.ReadAsStringAsync());
                _currentBearerToken = body["jwt"];
                _currentBearerTokenExpiration = DateTime.Parse(body["expiration"]);

                if (_debug)
                {
                    Console.WriteLine("bearer looks good");
                }
            }
            else
            {
                throw new FeatrixBadApiKeyError($"Failed to create authorization token from API Key: {response.StatusCode}. Are your client id and client secret correct?");
            }
        }

        private Dictionary<string, string> FeatrixHeaders(bool bearerGenerate = false, bool jsonRequest = false, Dictionary<string, string> extra = null)
        {
            var headers = new Dictionary<string, string>();

            if (jsonRequest)
            {
                headers.Add("Content-Type", "application/json");
                headers.Add("Accept", "text/plain");
                headers.Add("X-request-id", Guid.NewGuid().ToString());
                headers.Add("X-hostname", _hostname);
            }

            if (extra != null)
            {
                foreach (var entry in extra)
                {
                    headers[entry.Key] = entry.Value;
                }
            }

            if (!bearerGenerate)
            {
                if (_currentBearerToken == null || (_currentBearerTokenExpiration.HasValue && _currentBearerTokenExpiration < DateTime.Now))
                {
                    GenerateBearerTokenAsync().Wait();
                }

                if (_currentBearerToken == null)
                {
                    throw new FeatrixBadApiKeyError("Your ApiKey seems to have been invalidated, please create another one");
                }

                headers["Authorization"] = $"Bearer {_currentBearerToken}";
            }

            return headers;
        }

        private string ValidateUrl(string url, bool allowUnencryptedHttp)
        {
            if (string.IsNullOrEmpty(url))
            {
                throw new Exception("url argument must be in the form of https://host:port");
            }

            url = url.TrimEnd('/');
            ValidateUrlScheme(url, allowUnencryptedHttp);
            return $"{url}/api";
        }

        private void ValidateUrlScheme(string url, bool allowUnencryptedHttp)
        {
            if (!url.StartsWith("http://") && !url.StartsWith("https://"))
            {
                throw new Exception("url argument must be in the form of https://host:port");
            }
            else if (url.StartsWith("http://"))
            {
                ValidateLocalhostUrl(url, allowUnencryptedHttp);
            }
        }

        private void ValidateLocalhostUrl(string url, bool allowUnencryptedHttp)
        {
            if (!allowUnencryptedHttp)
            {
                var localhostPrefixes = new[] { "localhost", "127.0.0.1", "::1" };
                foreach (var lh in localhostPrefixes)
                {
                    if (url == $"http://{lh}" || url.StartsWith($"http://{lh}/") || url.StartsWith($"http://{lh}:"))
                    {
                        return;
                    }
                }

                throw new Exception("Non-HTTPS only supported for localhost without setting `allow_unencrypted_http`");
            }
        }

        public async Task<Dictionary<string, object>> PredictAsync(string neuralFunctionId, List<Dictionary<string, object>> query)
        {
            var url = $"{_url}/neural/models/prediction";
            var headers = FeatrixHeaders();

            if (_debug)
            {
                Console.WriteLine($"HTTP: {url}");
                Console.WriteLine($"headers: {string.Join(", ", headers)}");
            }

            try
            {
                var payload = JsonSerializer.Serialize(new
                {
                    job_type = "model-prediction",
                    model_id = neuralFunctionId,
                    query = query
                });

                var request = new HttpRequestMessage(HttpMethod.Post, url)
                {
                    Content = new StringContent(payload, Encoding.UTF8, "application/json")
                };

                foreach (var header in headers)
                {
                    request.Headers.Add(header.Key, header.Value);
                }

                var response = await httpClient.SendAsync(request);
                if (_debug)
                {
                    Console.WriteLine($"Response status: {response.StatusCode}");
                }

                var responseBody = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize<Dictionary<string, object>>(responseBody);
            }
            catch (HttpRequestException e)
            {
                if (_debug)
                {
                    Console.WriteLine($"Response exception: {e.Message}");
                }
                throw new FeatrixConnectionError(url, e.Message);
            }
        }
    }
}
