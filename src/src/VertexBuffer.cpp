#include "VertexBuffer.h"

using namespace alkaid;

VertexBuffer::VertexBuffer(const Builder& builder) {
    
}

VertexBuffer* VertexBuffer::Builder::build() {
    return new VertexBuffer(*this);
}

