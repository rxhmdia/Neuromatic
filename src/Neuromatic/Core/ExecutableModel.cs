using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neuromatic.Core
{
    /// <summary>
    /// An executable version of a model
    /// </summary>
    public class ExecutableModel
    {
        private BackendFunction _trainFunction;
        private BackendFunction _predictFunction;
        private IEnumerable<ExecutableModelNode> _outputs;
        private IEnumerable<ExecutableModelNode> _targets;
        private IEnumerable<ExecutableModelNode> _inputs;
        private IEnumerable<ExecutableModelNode> _losses;

        /// <summary>
        /// Initializes a new instance of <see cref="ExecutableModel"/>
        /// </summary>
        /// <param name="trainFunction">Function to train the model</param>
        /// <param name="predictFunction">Function to make predictions with this model</param>
        /// <param name="inputs">Input variables</param>
        /// <param name="outputs">Output variables</param>
        /// <param name="targets">Target variables</param>
        /// <param name="losses">Loss functions</param>
        public ExecutableModel(BackendFunction trainFunction, BackendFunction predictFunction, IEnumerable<ExecutableModelNode> inputs, IEnumerable<ExecutableModelNode> outputs, IEnumerable<ExecutableModelNode> targets, IEnumerable<ExecutableModelNode> losses)
        {
            _trainFunction = trainFunction;
            _predictFunction = predictFunction;
            _outputs = outputs;
            _targets = targets;
            _inputs = inputs;
            _losses = losses;
        }

        /// <summary>
        /// Trains the model with a set of features and targets
        /// </summary>
        /// <param name="x">
        /// A dictionary with a mapping between model inputs and the matching input values
        /// </param>
        /// <param name="y">
        /// A dictionary with model outputs and the matching output values
        /// </param>
        /// <returns>The values for all of the losses defined for the model</returns>
        public IEnumerable<object> TrainMiniBatch(IEnumerable<object> features, IEnumerable<object> targets)
        {
            var outputs = _trainFunction.Execute(features.Concat(targets));
            return outputs.Skip(_outputs.Count()).Take(_losses.Count());
        }

        /// <summary>
        /// Trains the model with a set of features and targets.
        /// </summary>
        /// <remarks>This method should only be used when you can handle all data in memory.
        /// In all other cases use a data source instead.</remarks>
        /// <param name="features">Data set containing the features</param>
        /// <param name="targets">Data set containing the targets</param>
        /// <param name="epochs">Number of epochs to train for</param>
        /// <param name="batchSize">Batch size to use while training</param>
        public void Train(IEnumerable<object> features, IEnumerable<object> targets, int epochs = 1, int batchSize = 32)
        {
            int featureRowCount = features.Count();
            int targetRowCount = targets.Count();

            if(featureRowCount != targetRowCount)
            {
                throw new ArgumentException("The same number of rows for features and targets must be provided");
            }

            if(epochs < 1)
            {
                throw new ArgumentException("Epochs must be greater or equal to 1", "epochs");
            }

            if(batchSize < 1)
            {
                throw new ArgumentException("Specify a batch size greater or equal to 1");
            }

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                for(int batchOffset = 0; batchOffset < featureRowCount - batchSize; batchOffset += batchSize)
                {
                    var batchFeatures = features.Skip(batchOffset).Take(batchSize);
                    var batchTargets = targets.Skip(batchOffset).Take(batchSize);

                    _trainFunction.Execute(batchFeatures.Concat(batchTargets));
                }
            }
        } 

        /// <summary>
        /// Generates output predictions based on the provided inputs
        /// </summary>
        /// <param name="features">
        /// A dictionary with a mapping between model inputs and the inputs for those model inputs
        /// </param>
        /// <returns>
        /// Returns a dictionary with a mapping between model outputs and the value for those model outputs
        /// </returns>
        public IEnumerable<object> Predict(IEnumerable<object> features)
        {
            return _predictFunction.Execute(features).Select(x => (float[][])x);
        }
    }
}
