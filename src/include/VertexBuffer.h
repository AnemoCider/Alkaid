#pragma once

namespace alkaid {

class VertexBuffer {

public:
    class Builder {
    public:
        Builder() = default;
        Builder& attribute();
        VertexBuffer* build();
    };

    VertexBuffer(const Builder& builder);

};

}

