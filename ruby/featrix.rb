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

require 'net/http'
require 'json'
require 'socket'
require 'securerandom'
require 'uri'

# Custom exceptions for Featrix errors
class FeatrixConnectionError < StandardError
  def initialize(url, message)
    super("Connection error for URL #{url}: #{message}")
  end
end

class FeatrixBadApiKeyError < StandardError
  def initialize(message)
    super(message)
  end
end

# Featrix class
class Featrix
  attr_accessor :debug, :client_id, :client_secret, :url, :hostname

  def initialize(
    url: "https://app.featrix.com",
    client_id: nil,
    client_secret: nil,
    allow_unencrypted_http: false,
    debug: false
  )
    @debug = debug
    @client_id = client_id
    @client_secret = client_secret
    @url = validate_url(url, allow_unencrypted_http)
    @hostname = Socket.gethostname
    @current_bearer_token = nil
    @current_bearer_token_expiration = nil

    generate_bearer_token

    raise "Your client id and client secret pair are invalid." if @current_bearer_token.nil?
  end

  # Generate Bearer Token
  def generate_bearer_token
    puts "_generate_bearer_token entered" if @debug

    headers = featrix_headers(bearer_generate: true)
    payload = {
      "client_id" => @client_id,
      "client_secret" => @client_secret
    }.to_json

    uri = URI("#{@url}/mosaic/keyauth/jwt")
    response = http_post_request(uri, headers, payload)

    if response.code.to_i == 200
      body = JSON.parse(response.body)
      @current_bearer_token = body["jwt"]
      @current_bearer_token_expiration = DateTime.iso8601(body["expiration"])

      puts "bearer looks good" if @debug
    else
      raise FeatrixBadApiKeyError.new("Failed to create authorization token from API Key: #{response.code}. Are your client id and client secret correct?")
    end
  end

  # Predict function
  def predict(neural_function_id, query)
    query = [query] if query.is_a?(Hash)

    headers = featrix_headers
    uri = URI("#{@url}/neural/models/prediction")

    puts "HTTP: #{uri}" if @debug
    puts "headers: #{headers}" if @debug

    payload = {
      "job_type" => "model-prediction",
      "model_id" => neural_function_id,
      "query" => query
    }.to_json

    begin
      response = http_post_request(uri, headers, payload)

      puts "Response status: #{response.code}" if @debug

      if response.code.to_i == 200
        JSON.parse(response.body)
      else
        raise FeatrixConnectionError.new(uri, "Unexpected response code: #{response.code}")
      end

    rescue Net::OpenTimeout, Net::ReadTimeout
      puts "Response exception: Timeout" if @debug
      retry
    rescue StandardError => e
      raise FeatrixConnectionError.new(uri, e.message)
    end
  end

  private

  # Helper to make HTTP POST requests
  def http_post_request(uri, headers, payload)
    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = uri.scheme == 'https'

    request = Net::HTTP::Post.new(uri.path, headers)
    request.body = payload

    http.request(request)
  end

  # Generate headers
  def featrix_headers(bearer_generate: false, json_request: false, extra: {})
    headers = {}

    if json_request
      headers.merge!({
        "Content-Type" => "application/json",
        "Accept" => "text/plain",
        "X-request-id" => SecureRandom.uuid,
        "X-hostname" => @hostname
      })
    end

    headers.merge!(extra)

    unless bearer_generate
      if @current_bearer_token.nil? || @current_bearer_token_expiration && @current_bearer_token_expiration < DateTime.now
        generate_bearer_token
      end

      if @current_bearer_token.nil?
        raise FeatrixBadApiKeyError.new("Your API Key seems to have been invalidated, please create another one.")
      end

      headers["Authorization"] = "Bearer #{@current_bearer_token}"
    end

    headers
  end

  # Validate URL
  def validate_url(url, allow_unencrypted_http)
    raise "URL must be provided in the form of https://host:port" if url.nil?

    url = url.chomp('/')
    validate_url_scheme(url, allow_unencrypted_http)
    "#{url}/api"
  end

  # Validate URL Scheme (HTTP/HTTPS)
  def validate_url_scheme(url, allow_unencrypted_http)
    unless url.start_with?("http://") || url.start_with?("https://")
      raise "URL must be in the form of https://host:port"
    end

    if url.start_with?("http://")
      validate_localhost_url(url, allow_unencrypted_http)
    end
  end

  # Validate Localhost for HTTP
  def validate_localhost_url(url, allow_unencrypted_http)
    unless allow_unencrypted_http
      localhost_prefixes = ["localhost", "127.0.0.1", "::1"]
      unless localhost_prefixes.any? { |lh| url == "http://#{lh}" || url.start_with?("http://#{lh}/") || url.start_with?("http://#{lh}:") }
        raise "Non-HTTPS only supported for localhost without setting allow_unencrypted_http."
      end
    end
  end
end
