<?php
/**
 * #############################################################################
 *
 *  Copyright (c) 2024, Featrix, Inc.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 * #############################################################################
 *
 *  Welcome to...
 *
 *   _______ _______ _______ _______ ______ _______ ___ ___
 *  |    ___|    ___|   _   |_     _|   __ \_     _|   |   |
 *  |    ___|    ___|       | |   | |      <_|   |_|-     -|
 *  |___|   |_______|___|___| |___| |___|__|_______|___|___|
 *
 *                                                Let's embed!
 *
 * #############################################################################
 *
 *  Sign up for Featrix at https://app.featrix.com/
 * 
 * #############################################################################
 *
 *  Check out the docs -- you can either call the python built-in help()
 *  or fire up your browser:
 *
 *     https://featrix-docs.readthedocs.io/en/latest/
 *
 *  You can also join our community Slack:
 *
 *     https://bits.featrix.com/slack
 *
 *  We'd love to hear from you: bugs, features, questions -- send them along!
 *
 *     hello@featrix.ai
 *
 * #############################################################################
 */

class FeatrixConnectionError extends Exception {
    public function __construct($url, $message) {
        parent::__construct("Connection error for URL $url: __$message__");
    }
}

class FeatrixBadApiKeyError extends Exception {
    public function __construct($msg) {
        $this->message = $msg;
    }
}

class Featrix {
    private $debug;
    private $clientId;
    private $clientSecret;
    private $url;
    private $hostname;
    private $currentBearerToken = null;
    private $currentBearerTokenExpiration = null;

    public function __construct(
        $url = "https://app.featrix.com",
        $clientId = null,
        $clientSecret = null,
        $allowUnencryptedHttp = false,
        $debug = false
    ) {
        $this->debug = $debug;
        $this->clientId = $clientId;
        $this->clientSecret = $clientSecret;
        $this->url = $this->validateUrl($url, $allowUnencryptedHttp);
        $this->hostname = gethostname();
        $this->generateBearerToken();
        if ($this->currentBearerToken === null) {
            throw new Exception("Your client id and client secret pair are invalid.");
        }
    }

    private function generateBearerToken() {
        if ($this->debug) {
            echo "_generate_bearer_token entered\n";
        }

        $headers = $this->featrixHeaders(true);

        $payload = json_encode([
            "client_id" => $this->clientId,
            "client_secret" => $this->clientSecret,
        ]);

        $ch = curl_init("{$this->url}/mosaic/keyauth/jwt");
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($httpCode == 200) {
            $body = json_decode($response, true);
            $this->currentBearerToken = $body["jwt"];
            $this->currentBearerTokenExpiration = new DateTime($body["expiration"]);
            if ($this->debug) {
                echo "bearer looks good\n";
            }
        } else {
            throw new FeatrixBadApiKeyError(
                "Failed to create authorization token from API Key: $httpCode. Are your client id and client secret correct?"
            );
        }
    }

    private function featrixHeaders($bearerGenerate = false, $jsonRequest = false, $extra = []) {
        $headers = [];

        if ($jsonRequest) {
            $headers = array_merge($headers, [
                "Content-Type: application/json",
                "Accept: text/plain",
                "X-request-id: " . uniqid(),
                "X-hostname: " . $this->hostname,
            ]);
        }

        if (!$bearerGenerate) {
            if ($this->currentBearerToken === null || ($this->currentBearerTokenExpiration && $this->currentBearerTokenExpiration < new DateTime())) {
                $this->generateBearerToken();
            }

            if ($this->currentBearerToken === null) {
                throw new FeatrixBadApiKeyError("Your ApiKey seems to have been invalidated, please create another one");
            }

            $headers[] = "Authorization: Bearer {$this->currentBearerToken}";
        }

        foreach ($extra as $key => $value) {
            $headers[] = "$key: $value";
        }

        return $headers;
    }

    private function validateUrl($url, $allowUnencryptedHttp) {
        if ($url === null) {
            throw new RuntimeException("url argument must be in the form of https://host:port");
        }

        $url = rtrim($url, "/");
        $this->validateUrlScheme($url, $allowUnencryptedHttp);
        return "$url/api";
    }

    private function validateUrlScheme($url, $allowUnencryptedHttp) {
        if (strpos($url, "http://") !== 0 && strpos($url, "https://") !== 0) {
            throw new RuntimeException("url argument must be in the form of https://host:port");
        } elseif (strpos($url, "http://") === 0) {
            $this->validateLocalhostUrl($url, $allowUnencryptedHttp);
        }
    }

    private function validateLocalhostUrl($url, $allowUnencryptedHttp) {
        if (!$allowUnencryptedHttp) {
            $localhostPrefixes = ["localhost", "127.0.0.1", "::1"];
            foreach ($localhostPrefixes as $lh) {
                $lh = "http://$lh";
                if ($url === $lh || strpos($url, "$lh/") === 0 || strpos($url, "$lh:") === 0) {
                    return;
                }
            }

            throw new RuntimeException("Non-HTTPS only supported for localhost without setting `allow_unencrypted_http`");
        }
    }

    public function predict($neuralFunctionId, $query) {
        if (!is_array($query)) {
            $query = [$query];
        }

        $url = "{$this->url}/neural/models/prediction";
        $headers = $this->featrixHeaders();
        $payload = json_encode([
            "job_type" => "model-prediction",
            "model_id" => $neuralFunctionId,
            "query" => $query,
        ]);

        if ($this->debug) {
            echo "HTTP: $url\n";
            echo "headers: " . json_encode($headers) . "\n";
        }

        $ch = curl_init($url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($this->debug) {
            echo "Response status: $httpCode\n";
        }

        if ($httpCode == 200) {
            return json_decode($response, true);
        }

        return null;
    }
}
?>
