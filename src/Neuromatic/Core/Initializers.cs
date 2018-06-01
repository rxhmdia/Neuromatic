using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Core
{
    /// <summary>
    /// Initializer functions are used to initialize weights and biases in the model.
    /// </summary>
    public abstract class Initializers
    {
        /// <summary>
        /// Creates an initializer that initializes a set of weights with a normal distribution
        /// </summary>
        /// <param name="mean">Mean of the distribution (default 0.0)</param>
        /// <param name="standardDeviation">Standard distribution (default 0.05)</param>
        /// <param name="seed">The random seed</param>
        /// <returns>Returns the new initializer</returns>
        public abstract InitializationFunction RandomNormal(float mean = 0.0f, float standardDeviation = 0.05f, float? seed = null);
    }
}
