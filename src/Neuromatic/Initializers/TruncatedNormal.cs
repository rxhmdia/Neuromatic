using System;
using TensorFlow;

namespace Neuromatic.Initializers
{
    /// <summary>
    /// The truncated normal initializer is similar to the <see cref="RandomNormal"/> initializer.
    /// It draws random values from a random normal distribution. The difference with the <see cref="RandomNormal"/>
    /// initializer is that all values above and below 2*standard-deviation are redrawn.
    /// </summary>
    public class TruncatedNormal: InitializationFunction
    {
        private readonly double _mean;
        private readonly double _standardDeviation;
        private readonly int? _seed;

        /// <summary>
        /// Initializes a new instance of <see cref="TruncatedNormal"/>
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="standardDeviation"></param>
        /// <param name="seed"></param>
        public TruncatedNormal(double mean = 0.0, double standardDeviation = 1.0, int? seed = null)
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
            return graph.ParameterizedTruncatedNormal(graph.Const(shape), 
                graph.Const(_mean), graph.Const(_standardDeviation),
                graph.Const(Double.MinValue), graph.Const(double.MaxValue), _seed);
        }
    }
}