export module vertexbuffer;

namespace alkaid {

export class VertexBuffer {

public:
    class Builder {
    public:
        Builder() = default;
        Builder& attribute();
        VertexBuffer* build();
    };

    VertexBuffer(const Builder& builder);

};

VertexBuffer::VertexBuffer(const Builder& builder) {
    
}

VertexBuffer* VertexBuffer::Builder::build() {
    return new VertexBuffer(*this);
}

}
