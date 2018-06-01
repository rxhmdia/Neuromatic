using Neuromatic.Core;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace Neuromatic.TensorFlow
{
    /// <summary>
    /// A tensorflow specific implementation of an executable model
    /// </summary>
    public class TensorflowExecutableModel : ExecutableModel
    {
        /// <summary>
        /// Generates output predictions based on the provided inputs
        /// </summary>
        /// <param name="features">
        /// A dictionary with a mapping between model inputs and the inputs for those model inputs
        /// </param>
        /// <returns>
        /// Returns a dictionary with a mapping between model outputs and the value for those model outputs
        /// </returns>
        public override Dictionary<string, float[][]> Predict(Dictionary<string, float[][]> features)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Scores the model using the metrics that were defined for the model.
        /// </summary>
        /// <returns>
        /// Returns a dictionary where the key is the name of the
        /// metric and the value is the value for that metric.
        /// </returns>
        public override Dictionary<string, float> Score()
        {
            throw new NotImplementedException();
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
        public override void Train(Dictionary<string, float[][]> features, Dictionary<string, float[][]> targets)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public override void Dispose()
        {

        }
    }
}
