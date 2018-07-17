using Neuromatic.Layers;
using Neuromatic.Losses;
using Neuromatic.Optimizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorFlow;

namespace Neuromatic
{
    /// <summary>
    /// A model is defined as a set of inputs and a set of outputs connected to the provided inputs.
    /// To train a model you first compile the model. Then you use one of the training methods to
    /// optimize the parameters of the model. You can make predictions using one of the prediction methods.
    /// </summary>
    public class Model
    {
        private IEnumerable<Input> _inputs;
        private IEnumerable<Layer> _outputs;

        private Dictionary<Input, TFOutput> _inputMapping;
        private Dictionary<Layer, TFOutput> _outputMapping;
        private Dictionary<Layer, TFOutput> _placeholderMapping;

        private TFOutput _modelLoss;
        private TFSession _session;
        private TFGraph _graph;
        private IEnumerable<TFOutput> _parameters;
        private IEnumerable<TFOperation> _initializers;

        private Optimizer _optimizer;

        /// <summary>
        /// Initializes a new instance of <see cref="Model"/>
        /// </summary>
        /// <param name="inputs">Input layers for the model</param>
        /// <param name="outputs">Output layers for the model that are connected to the input layers</param>
        public Model(IEnumerable<Input> inputs, IEnumerable<Layer> outputs)
        {
            _inputs = inputs;
            _outputs = outputs;
        }

        /// <summary>
        /// Compiles the model into an executable graph
        /// </summary>
        /// <param name="optimizer">The optimization algorithm to use for training the model</param>
        /// <param name="losses">The losses for each of the outputs of the model</param>
        /// <remarks>The list of loss functions should be in order of the outputs of the model</remarks>
        public void Compile(Optimizer optimizer, IEnumerable<LossFunction> losses)
        {
            if (optimizer == null)
            {
                throw new ArgumentNullException(
                    "The optimizer must be specified",
                    nameof(optimizer));
            }

            if (losses.Count() != _outputs.Count())
            {
                throw new ArgumentException(
                    "The number of loss functions does not match the number of outputs of the model",
                    nameof(losses));
            }

            _graph = new TFGraph();

            var compilationContext = new ModelCompilationContext(_graph);

            _optimizer = optimizer;
            _inputMapping = new Dictionary<Input, TFOutput>();
            _outputMapping = new Dictionary<Layer, TFOutput>();
            _placeholderMapping = new Dictionary<Layer, TFOutput>();

            var compiledLosses = new List<TFOutput>();

            var layersWithLosses = Enumerable.Zip(_outputs, losses, (layer, loss) => (layer, loss));

            // By compiling the outputs, the layers that are connected
            // to the outputs are also compiled. This goes all the way back to the inputs.
            foreach (var (layer, loss) in layersWithLosses)
            {
                var placeholder = _graph.Placeholder(TFDataType.Double, new TFShape(layer.OutputShape));
                var output = layer.Compile(compilationContext);

                _outputMapping.Add(layer, output);
                _placeholderMapping.Add(layer, placeholder);

                var compiledLoss = loss.Compile(compilationContext, output, placeholder);

                compiledLosses.Add(compiledLoss);
            }

            foreach (var input in _inputs)
            {
                _inputMapping.Add(input, input.Configuration.Output);
            }

            _modelLoss = compiledLosses.Aggregate((left, right) => _graph.Add(left, right));
            _optimizer.Compile(_graph, _modelLoss, compilationContext.Parameters);

            _initializers = compilationContext.Initializers;
            _parameters = compilationContext.Parameters;
        }

        /// <summary>
        /// Trains the model with a single minibatch of data
        /// </summary>
        /// <param name="features">Features for the batch</param>
        /// <param name="targets">Targets for the batch</param>
        public void TrainMinibatch(Dictionary<Input, Array> features, Dictionary<Layer, Array> targets)
        {
            if (_graph == null)
            {
                throw new InvalidOperationException("The model must be compiled first before you can train it.");
            }

            if (!features.Select(x => _inputMapping.ContainsKey(x.Key)).All(x => x))
            {
                throw new ArgumentException("Not all inputs have a value provided for them. Please make sure that you provide data for each of the inputs of the model.");
            }

            if (!targets.Select(x => _placeholderMapping.ContainsKey(x.Key)).All(x => x))
            {
                throw new ArgumentException("Not all targets have a value provided for them. Please make sure that you provide data for each of the targets of the model.");
            }

            EnsureSession();

            var modelInputs = new Dictionary<TFOutput, Array>();

            foreach (var keyValuePair in features)
            {
                modelInputs.Add(_inputMapping[keyValuePair.Key], keyValuePair.Value);
            }

            foreach (var keyValuePair in targets)
            {
                modelInputs.Add(_placeholderMapping[keyValuePair.Key], keyValuePair.Value);
            }

            _optimizer.Execute(_session, modelInputs);
        }

        /// <summary>
        /// Makes a prediction using the model
        /// </summary>
        /// <param name="features">Input features to use for making the prediction</param>
        /// <returns>Returns the values for each of the outputs of the model</returns>
        public Dictionary<Layer, Array> Predict(Dictionary<Input, Array> features)
        {
            EnsureSession();

            var results = new Dictionary<Layer,Array>();
            var runner = _session.GetRunner();

            foreach (var keyValuePair in features)
            {
                runner.AddInput(_inputMapping[keyValuePair.Key], keyValuePair.Value);   
            }

            foreach(var keyValuePair in _outputMapping)
            {
                var outputValue = runner.Run(keyValuePair.Value);
                results.Add(keyValuePair.Key, (Array)outputValue.GetValue());
            }

            return results;
        }

        private void EnsureSession()
        {
            if (_session == null)
            {
                _session = new TFSession(_graph);

                // Initializers are run upon session creation.
                // This ensures that all variables are initialized when we try to train them first time.
                _session.GetRunner()
                    .AddTarget(_initializers.ToArray())
                    .Run();
            }
        }
    }
}
