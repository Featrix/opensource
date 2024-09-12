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

import Foundation

// MARK: - Custom Errors

enum FeatrixError: Error {
    case connectionError(url: String, message: String)
    case badApiKeyError(message: String)
}

// MARK: - Featrix Class

class Featrix {
    
    // MARK: - Properties
    
    private var url: String
    private var clientId: String?
    private var clientSecret: String?
    private var currentBearerToken: String?
    private var currentBearerTokenExpiration: Date?
    private var hostname: String
    private var debug: Bool
    
    // MARK: - Initializer
    
    init(url: String = "https://app.featrix.com", clientId: String? = nil, clientSecret: String? = nil, allowUnencryptedHttp: Bool = false, debug: Bool = false) throws {
        self.url = Featrix.validateUrl(url: url, allowUnencryptedHttp: allowUnencryptedHttp)
        self.clientId = clientId
        self.clientSecret = clientSecret
        self.hostname = ProcessInfo.processInfo.hostName
        self.debug = debug
        
        try generateBearerToken()
        
        guard currentBearerToken != nil else {
            throw FeatrixError.badApiKeyError(message: "Your client id and client secret pair are invalid.")
        }
    }
    
    // MARK: - Private Methods
    
    private func generateBearerToken() throws {
        if debug {
            print("_generate_bearer_token entered")
        }
        
        let headers = featrixHeaders(bearerGenerate: true)
        
        let payload: [String: Any] = [
            "client_id": clientId ?? "",
            "client_secret": clientSecret ?? ""
        ]
        
        let jsonData = try JSONSerialization.data(withJSONObject: payload)
        var request = URLRequest(url: URL(string: "\(url)/mosaic/keyauth/jwt")!)
        request.httpMethod = "POST"
        request.allHTTPHeaderFields = headers
        request.httpBody = jsonData
        
        let (data, response) = try await URLSession.shared.data(for: request)
        let httpResponse = response as! HTTPURLResponse
        
        if httpResponse.statusCode == 200 {
            let json = try JSONSerialization.jsonObject(with: data, options: []) as! [String: Any]
            currentBearerToken = json["jwt"] as? String
            if let expiration = json["expiration"] as? String {
                currentBearerTokenExpiration = ISO8601DateFormatter().date(from: expiration)
            }
            if debug {
                print("bearer looks good")
            }
        } else {
            throw FeatrixError.badApiKeyError(message: "Failed to create authorization token from API Key: \(httpResponse.statusCode). Are your client id and client secret correct?")
        }
    }
    
    private func featrixHeaders(bearerGenerate: Bool = false, jsonRequest: Bool = false) -> [String: String] {
        var headers: [String: String] = [:]
        
        if jsonRequest {
            headers = [
                "Content-type": "application/json",
                "Accept": "text/plain",
                "X-request-id": UUID().uuidString,
                "X-hostname": hostname
            ]
        }
        
        if !bearerGenerate {
            if currentBearerToken == nil || (currentBearerTokenExpiration != nil && currentBearerTokenExpiration! < Date()) {
                try? generateBearerToken()
            }
            
            guard let token = currentBearerToken else {
                fatalError("API Key seems to have been invalidated, please create another one")
            }
            headers["Authorization"] = "Bearer \(token)"
        }
        
        return headers
    }
    
    private static func validateUrl(url: String, allowUnencryptedHttp: Bool) -> String {
        let trimmedUrl = url.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        validateUrlScheme(url: trimmedUrl, allowUnencryptedHttp: allowUnencryptedHttp)
        return "\(trimmedUrl)/api"
    }
    
    private static func validateUrlScheme(url: String, allowUnencryptedHttp: Bool) {
        if !(url.hasPrefix("http://") || url.hasPrefix("https://")) {
            fatalError("URL must be in the form of https://host:port")
        } else if url.hasPrefix("http://") {
            validateLocalhostUrl(url: url, allowUnencryptedHttp: allowUnencryptedHttp)
        }
    }
    
    private static func validateLocalhostUrl(url: String, allowUnencryptedHttp: Bool) {
        if !allowUnencryptedHttp {
            let localhostPrefixes = ["localhost", "127.0.0.1", "::1"]
            for lh in localhostPrefixes {
                let httpLocalhost = "http://\(lh)"
                if url == httpLocalhost || url.hasPrefix(httpLocalhost + "/") || url.hasPrefix(httpLocalhost + ":") {
                    return
                }
            }
            fatalError("Non-HTTPS only supported for localhost without setting `allowUnencryptedHttp`")
        }
    }
    
    // MARK: - Public Methods
    
    func predict(neuralFunctionId: String, query: [[String: Any]]) async throws -> [String: Any] {
        let predictionUrl = "\(url)/neural/models/prediction"
        let headers = featrixHeaders()
        
        if debug {
            print("HTTP: \(predictionUrl)")
            print("headers: \(headers)")
        }
        
        let payload: [String: Any] = [
            "job_type": "model-prediction",
            "model_id": neuralFunctionId,
            "query": query
        ]
        
        let jsonData = try JSONSerialization.data(withJSONObject: payload)
        var request = URLRequest(url: URL(string: predictionUrl)!)
        request.httpMethod = "POST"
        request.allHTTPHeaderFields = headers
        request.httpBody = jsonData
        
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let httpResponse = response as! HTTPURLResponse
            
            if debug {
                print("Response status: \(httpResponse.statusCode)")
            }
            
            if httpResponse.statusCode == 200 {
                let jsonResponse = try JSONSerialization.jsonObject(with: data, options: []) as! [String: Any]
                return jsonResponse
            } else {
                throw FeatrixError.connectionError(url: predictionUrl, message: "Unexpected HTTP status: \(httpResponse.statusCode)")
            }
        } catch let error {
            throw FeatrixError.connectionError(url: predictionUrl, message: error.localizedDescription)
        }
    }
}
