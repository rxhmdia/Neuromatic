using Neuromatic.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.TensorFlow
{
    public class TensorFlowActivations : Activations
    {
        private TensorFlowModelBackend _backend;

        public TensorFlowActivations(TensorFlowModelBackend backend)
        {
            _backend = backend;
        }

        public override ActivationFunction Sigmoid()
        {
            throw new NotImplementedException();
        }
    }
}
