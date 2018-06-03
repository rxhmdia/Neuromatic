using Neuromatic.Initializers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic
{
    /// <summary>
    /// Defines a set of standard initializers available in the framework
    /// </summary>
    public static class StandardInitializers
    {
        /// <summary>
        /// Defines an initialization function that performs a random initialization with a normal distribution
        /// </summary>
        /// <returns>Returns the initialization function</returns>
        public static InitializationFunction RandomNormal(float mean = 0.0f, float standardDeviation = 0.05f, int? seed = null)
        {
            return new RandomNormal(mean, standardDeviation, seed);
        }
    }
}
