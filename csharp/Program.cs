using System.Text.Json;
using FeatrixExample;

var clientId = "<fill this in>";
var clientSecret = "<fill this in>";
var url = "<fill this in>";
var neuralFunctionId = "<fill this in>";

try {
    // Create instance of Featrix
    var featrix = await Featrix.CreateAsync(clientId, clientSecret, url, true, true);
    
    // Run prediction
    var prediction = await featrix.PredictAsync(neuralFunctionId, new List<Dictionary<string, object>>());
    
    var json = JsonSerializer.Serialize(prediction);
    Console.WriteLine("Prediction returned: " + json);
}
catch(Exception ex) {
    Console.Error.WriteLine("Unhandled exception: " + ex.ToString());
}