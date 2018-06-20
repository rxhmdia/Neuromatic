using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace Neuromatic.Initializers
{
    /// <summary>
    /// Initializer function that generates a tensor with random normally distributed set of numbers.
    /// </summary>
    public class RandomNormal : InitializationFunction
    {
        private double _mean;
        private double _standardDeviation;
        private int? _seed;

        /// <summary>
        /// Initializes a new instance of <see cref="RandomNormal"/>
        /// </summary>
        /// <param name="mean">Mean of the distribution</param>
        /// <param name="standardDeviation">Standard deviation of the distribution</param>
        /// <param name="seed">Random seed</param>
        public RandomNormal(double mean = 0.0, double standardDeviation = 0.05, int? seed = null)
        {
            _mean = mean;
            _standardDeviation = standardDeviation;
            _seed = seed;
        }

        /// <summary>
        /// Compiles the initialization function
        /// </summary>
        /// <param name="graph">Tensorflow graph to use for creating the initialization function</param>
        /// <param name="shape">The shape of the variable to initialize</param>
        /// <returns>Returns the compiled initialization function</returns>
        public override TFOutput Compile(TFGraph graph, TFShape shape)
        {
            return graph.RandomNormal(shape, _mean, _standardDeviation, _seed);
        }
    }
}
