using System;
using System.Collections.Generic;
using System.Text;
using Neuromatic.Core;

namespace Neuromatic.Initializers
{
    /// <summary>
    /// Initializes variables with a normal distributed set of values
    /// </summary>
    public class RandomNormal : InitializationFunction
    {
        /// <summary>
        /// Initializes a new instance of <see cref="RandomNormal"/>
        /// </summary>
        /// <param name="mean">Mean of the distribution</param>
        /// <param name="standardDeviation">Standard deviation of the distribution</param>
        /// <param name="seed">Random seed to use for the random number generator</param>
        public RandomNormal(float mean, float standardDeviation, int? seed)
        {
            Seed = seed;
            StandardDeviation = standardDeviation;
            Mean = mean;
        }

        /// <summary>
        /// Mean of the distribution
        /// </summary>
        public float Mean { get; set; }

        /// <summary>
        /// Standard deviation to use for the distribution
        /// </summary>
        public float StandardDeviation { get; set; }
        
        /// <summary>
        /// Random seed used to initialize the function
        /// </summary>
        public int? Seed { get; set; }

        /// <summary>
        /// Compiles the initialization function
        /// </summary>
        /// <param name="shape">Shape of the variable to initialize</param>
        /// <param name="backend">Backend to use for compilation</param>
        /// <returns></returns>
        public override ExecutableModelNode Compile(long[] shape, ModelBackend backend)
        {
            return backend.Float(backend.RandomNormal(shape, Mean, StandardDeviation, Seed));
        }
    }
}
