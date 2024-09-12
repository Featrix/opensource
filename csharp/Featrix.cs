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
using System.ComponentModel.DataAnnotations;
using System.Net;

namespace FeatrixExample;

public class FeatrixConnectionError : Exception
{
    public FeatrixConnectionError(string url, string message)
        : base($"Connection error for URL {url}: __{message}__") { }
    public FeatrixConnectionError(string url, string message, Exception inner) 
        : base($"Connection error for URL {url}: __{message}__", inner) {}
}

public class FeatrixBadApiKeyError : Exception
{
    public FeatrixBadApiKeyError(string message) : base(message) { }
}

internal class Requires {
    public static void NotDefaultOrEmptyOrWhitepsace(string name, string v) {
        if(string.IsNullOrWhiteSpace(v))
            throw new ArgumentException(name + " is either null, emtpy, or comprised entirely of whitespace");
    }
}

internal class FeatrixCaller {
    private const string DEFAULT_URL = "https://app.featrix.com";
    private string _clientId;
    private string _clientSecret;
    private bool _debug = false;
    private bool _hasInit = false;
    private string _sourceUrl;
    private string? _url;
    private bool _allowUnencryptedHttp;
    private string? _hostname;
    private string? _currentBearerToken;
    private DateTime? _currentBearerTokenExpiration;

    public FeatrixCaller(bool debug, string clientId, string clientSecret, bool allowUnencryptedHttp, string? sourceUrl = null) {
        Requires.NotDefaultOrEmptyOrWhitepsace(nameof(clientId), clientId);
        Requires.NotDefaultOrEmptyOrWhitepsace(nameof(clientSecret), clientSecret);
        this._sourceUrl = sourceUrl ?? DEFAULT_URL;
        this._allowUnencryptedHttp = allowUnencryptedHttp;
        this._clientId = clientId;
        this._clientSecret = clientSecret;
        this._debug = debug;
    }

    public HttpClient HttpClient { get; } = new ();

    public string Url { 
        get {
            if(!this._hasInit)
                throw new InvalidOperationException("attempt to access before init");
            return this._url!;
        }
        set {
            this._url = value;
        }
    }

    public async Task InitAsync() {
        if(this._hasInit)
            throw new InvalidOperationException("attemp to re-init");
        
        _hostname = Dns.GetHostName();
        Url = CreateUrl(this._sourceUrl, this._allowUnencryptedHttp);
        await GenerateBearerTokenAsync();
        this._hasInit = true;
    }

    public async Task<Dictionary<string, string>> GetHeadersAsync(bool bearerGenerate = false, bool jsonRequest = false, Dictionary<string, string>? extra = null)
    {
        var headers = new Dictionary<string, string>();

        if (jsonRequest)
        {
            headers.Add("Content-Type", "application/json");
            headers.Add("Accept", "text/plain");
            headers.Add("X-request-id", Guid.NewGuid().ToString());
            headers.Add("X-hostname", _hostname!);
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
                await GenerateBearerTokenAsync();
            }

            if (_currentBearerToken == null)
            {
                throw new FeatrixBadApiKeyError("Your ApiKey seems to have been invalidated, please create another one");
            }

            headers["Authorization"] = $"Bearer {_currentBearerToken}";
        }

        return headers;
    }

    private static string CreateUrl(string sourceUrl, bool allowUnencryptedHttp)
    {
        if (string.IsNullOrEmpty(sourceUrl))
            throw new Exception($"url argument must be in the form of http${(allowUnencryptedHttp ? "" : "s")}://host:port");
        sourceUrl = sourceUrl.TrimEnd('/');
        ValidateUrlScheme(sourceUrl, allowUnencryptedHttp);
        return $"{sourceUrl}/api";
    }

    private static void ValidateUrlScheme(string url, bool allowUnencryptedHttp)
    {
        if (!url.StartsWith("http://") && !url.StartsWith("https://"))
        {
            throw new ArgumentException("url argument must be in the form of https://host:port");
        }
        else if (url.StartsWith("http://"))
        {
            ValidateLocalhostUrl(url, allowUnencryptedHttp);
        }
    }

    private static void ValidateLocalhostUrl(string url, bool allowUnencryptedHttp)
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

            throw new ArgumentException("Non-HTTPS only supported for localhost without setting `allow_unencrypted_http`");
        }
    }

    private async Task GenerateBearerTokenAsync()
    {
        if (this._debug)
        {
            Console.WriteLine("_generate_bearer_token entered");
        }

        var headers = await this.GetHeadersAsync(bearerGenerate: true);

        var payload = JsonSerializer.Serialize(new
        {
            client_id = this._clientId,
            client_secret = this._clientSecret
        });

        var response = await this.HttpClient.PostAsync($"{Url}/mosaic/keyauth/jwt", new StringContent(payload, Encoding.UTF8, "application/json"));

        if (response.IsSuccessStatusCode)
        {
            var body = JsonSerializer.Deserialize<Dictionary<string, string>>(await response.Content.ReadAsStringAsync());
            if(body == null)
                throw new FeatrixBadApiKeyError("body was unable to deserialize");
            
            if(!DateTime.TryParse(body["expiration"], out var currentBearerTokenExpiration))
                throw new FeatrixBadApiKeyError("Failed to parse expiration DateTime");

            this._currentBearerToken = body["jwt"];
            this._currentBearerTokenExpiration = currentBearerTokenExpiration;

            if (this._debug)
            {
                Console.WriteLine("bearer looks good");
            }
        }
        else
        {
            throw new FeatrixBadApiKeyError($"Failed to create authorization token from API Key: {response.StatusCode}. Are your client id and client secret correct?");
        }
    }
}

public class Featrix
{
    private bool _debug;
    private readonly FeatrixCaller _caller;

    public async static Task<Featrix> CreateAsync(string clientId, string clientSecret, string? url = null, bool allowUnencryptedHttp = false, bool debug = false) {        
        var caller = new FeatrixCaller(debug, clientId, clientSecret, allowUnencryptedHttp, url);
        await caller.InitAsync();
        return new Featrix(debug, caller);
    }

    private Featrix(bool debug, FeatrixCaller caller)
    {
        this._debug = debug;
        this._caller = caller;
    }

    public async Task<Dictionary<string, object>> PredictAsync(string neuralFunctionId, List<Dictionary<string, object>> query)
    {
        var url = $"{this._caller.Url}/neural/models/prediction";
        var headers = await this._caller.GetHeadersAsync();

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

            var httpResponse = await this._caller.HttpClient.SendAsync(request);
            if (_debug)
            {
                Console.WriteLine($"Response status: {httpResponse.StatusCode}");
            }

            var response = JsonSerializer.Deserialize<Dictionary<string, object>>(await httpResponse.Content.ReadAsStringAsync());
            if(response == null) {
                throw new FeatrixConnectionError(url, "Unable to deserialize response into a dictionary");
            }

            return response;
        }
        catch (HttpRequestException e)
        {
            if (_debug)
            {
                Console.WriteLine($"Response exception: {e.Message}");
            }
            throw new FeatrixConnectionError(url, e.Message, e);
        }
    }
}