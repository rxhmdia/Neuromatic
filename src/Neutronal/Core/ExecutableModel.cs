using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Core
{
    /// <summary>
    /// An executable version of a model
    /// </summary>
    public abstract class ExecutableModel: IDisposable
    {
        /// <summary>
        /// Trains the model with a set of features and targets
        /// </summary>
        /// <param name="x">
        /// A dictionary with a mapping between model inputs and the matching input values
        /// </param>
        /// <param name="y">
        /// A dictionary with model outputs and the matching output values
        /// </param>
        public abstract void Train(Dictionary<string, float[][]> features, Dictionary<string, float[][]> targets);

        /// <summary>
        /// Generates output predictions based on the provided inputs
        /// </summary>
        /// <param name="features">
        /// A dictionary with a mapping between model inputs and the inputs for those model inputs
        /// </param>
        /// <returns>
        /// Returns a dictionary with a mapping between model outputs and the value for those model outputs
        /// </returns>
        public abstract Dictionary<string, float[][]> Predict(Dictionary<string, float[][]> features);

        /// <summary>
        /// Scores the model using the metrics that were defined for the model.
        /// </summary>
        /// <returns>
        /// Returns a dictionary where the key is the name of the 
        /// metric and the value is the value for that metric.
        /// </returns>
        public abstract Dictionary<string, float> Score();

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public abstract void Dispose();
    }
}
