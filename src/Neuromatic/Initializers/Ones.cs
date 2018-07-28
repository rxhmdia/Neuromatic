using TensorFlow;

namespace Neuromatic.Initializers
{
    /// <summary>
    /// This initialization function assigns ones according to the required shape
    /// </summary>
    public class Ones: InitializationFunction
    {
        /// <summary>
        /// Compiles the initialization function
        /// </summary>
        /// <param name="graph">Tensorflow graph to use for creating the initialization function</param>
        /// <param name="shape">The shape of the variable to initialize</param>
        /// <returns>Returns the compiled initialization function</returns>
        public override TFOutput Compile(TFGraph graph, TFShape shape)
        {
            return graph.Ones(shape);
        }
    }
}