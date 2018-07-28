using TensorFlow;

namespace Neuromatic.Layers
{
    public class Dropout: Layer
    {
        private readonly Layer _input;
        private readonly int? _seed;
        private readonly double _rate;

        public Dropout(double rate, Layer input, int? seed = null, string name = null) : base(name)
        {
            _input = input;
            _seed = seed;
            _rate = rate;
        }

        /// <summary>
        /// <para>
        /// Builds the layer by converting the abstract definition of the layer into 
        /// a concrete set of instructions for Tensorflow and a layer configuration
        /// for use when training the model.
        /// </para>
        /// <para>This method should register any parameters and initializers with the compilation context.
        /// So that they can be used during the training phase. </para>
        /// <para>Additionally you are required to store the layer configuration in the 
        /// <see cref="Layer.Configuration"/> property. This information is required as metadata 
        /// when the model is used.</para>
        /// <param name="context">Use this context to register trainable parameters
        /// and build the computational graph for the layer</param>
        public override TFOutput Compile(ModelCompilationContext context)
        {
            var inputLayer = _input.Compile(context);
            var keepProb = context.Graph.Const(_rate);

            var output = context.Graph.Dropout(inputLayer, keepProb, 
                context.Graph.GetTensorShape(inputLayer), _seed);

            Configuration = new LayerConfiguration(new TFOutput[] { }, new TFOperation[] { }, output);

            return output;
        }

        /// <summary>
        /// Gets the output shape for the layer
        /// </summary>
        public override long[] OutputShape => _input.OutputShape;
    }
}