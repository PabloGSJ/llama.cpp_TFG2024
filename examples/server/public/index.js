function t(){throw new Error("Cycle detected")}function n(){if(u>1){u--;return}let t,n=!1;while(void 0!==_){let i=_;_=void 0;f++;while(void 0!==i){const _=i.o;i.o=void 0;i.f&=-3;if(!(8&i.f)&&a(i))try{i.c()}catch(e){if(!n){t=e;n=!0}}i=_}}f=0;u--;if(n)throw t}function e(t){if(u>0)return t();u++;try{return t()}finally{n()}}let i,_,o=0;function r(t){if(o>0)return t();const n=i;i=void 0;o++;try{return t()}finally{o--;i=n}}let u=0,f=0,l=0;function s(t){if(void 0===i)return;let n=t.n;if(void 0===n||n.t!==i){n={i:0,S:t,p:i.s,n:void 0,t:i,e:void 0,x:void 0,r:n};if(void 0!==i.s)i.s.n=n;i.s=n;t.n=n;if(32&i.f)t.S(n);return n}else if(-1===n.i){n.i=0;if(void 0!==n.n){n.n.p=n.p;if(void 0!==n.p)n.p.n=n.n;n.p=i.s;n.n=void 0;i.s.n=n;i.s=n}return n}}function c(t){this.v=t;this.i=0;this.n=void 0;this.t=void 0}c.prototype.h=function(){return!0};c.prototype.S=function(t){if(this.t!==t&&void 0===t.e){t.x=this.t;if(void 0!==this.t)this.t.e=t;this.t=t}};c.prototype.U=function(t){if(void 0!==this.t){const n=t.e,e=t.x;if(void 0!==n){n.x=e;t.e=void 0}if(void 0!==e){e.e=n;t.x=void 0}if(t===this.t)this.t=e}};c.prototype.subscribe=function(t){const n=this;return S((function(){const e=n.value,i=32&this.f;this.f&=-33;try{t(e)}finally{this.f|=i}}))};c.prototype.valueOf=function(){return this.value};c.prototype.toString=function(){return this.value+""};c.prototype.toJSON=function(){return this.value};c.prototype.peek=function(){return this.v};Object.defineProperty(c.prototype,"value",{get(){const t=s(this);if(void 0!==t)t.i=this.i;return this.v},set(e){if(i instanceof v)!function(){throw new Error("Computed cannot have side-effects")}();if(e!==this.v){if(f>100)t();this.v=e;this.i++;l++;u++;try{for(let t=this.t;void 0!==t;t=t.x)t.t.N()}finally{n()}}}});function h(t){return new c(t)}function a(t){for(let n=t.s;void 0!==n;n=n.n)if(n.S.i!==n.i||!n.S.h()||n.S.i!==n.i)return!0;return!1}function p(t){for(let n=t.s;void 0!==n;n=n.n){const e=n.S.n;if(void 0!==e)n.r=e;n.S.n=n;n.i=-1;if(void 0===n.n){t.s=n;break}}}function d(t){let n,e=t.s;while(void 0!==e){const t=e.p;if(-1===e.i){e.S.U(e);if(void 0!==t)t.n=e.n;if(void 0!==e.n)e.n.p=t}else n=e;e.S.n=e.r;if(void 0!==e.r)e.r=void 0;e=t}t.s=n}function v(t){c.call(this,void 0);this.x=t;this.s=void 0;this.g=l-1;this.f=4}(v.prototype=new c).h=function(){this.f&=-3;if(1&this.f)return!1;if(32==(36&this.f))return!0;this.f&=-5;if(this.g===l)return!0;this.g=l;this.f|=1;if(this.i>0&&!a(this)){this.f&=-2;return!0}const t=i;try{p(this);i=this;const t=this.x();if(16&this.f||this.v!==t||0===this.i){this.v=t;this.f&=-17;this.i++}}catch(t){this.v=t;this.f|=16;this.i++}i=t;d(this);this.f&=-2;return!0};v.prototype.S=function(t){if(void 0===this.t){this.f|=36;for(let t=this.s;void 0!==t;t=t.n)t.S.S(t)}c.prototype.S.call(this,t)};v.prototype.U=function(t){if(void 0!==this.t){c.prototype.U.call(this,t);if(void 0===this.t){this.f&=-33;for(let t=this.s;void 0!==t;t=t.n)t.S.U(t)}}};v.prototype.N=function(){if(!(2&this.f)){this.f|=6;for(let t=this.t;void 0!==t;t=t.x)t.t.N()}};v.prototype.peek=function(){if(!this.h())t();if(16&this.f)throw this.v;return this.v};Object.defineProperty(v.prototype,"value",{get(){if(1&this.f)t();const n=s(this);this.h();if(void 0!==n)n.i=this.i;if(16&this.f)throw this.v;return this.v}});function y(t){return new v(t)}function m(t){const e=t.u;t.u=void 0;if("function"==typeof e){u++;const _=i;i=void 0;try{e()}catch(n){t.f&=-2;t.f|=8;g(t);throw n}finally{i=_;n()}}}function g(t){for(let n=t.s;void 0!==n;n=n.n)n.S.U(n);t.x=void 0;t.s=void 0;m(t)}function b(t){if(i!==this)throw new Error("Out-of-order effect");d(this);i=t;this.f&=-2;if(8&this.f)g(this);n()}function k(t){this.x=t;this.u=void 0;this.s=void 0;this.o=void 0;this.f=32}k.prototype.c=function(){const t=this.S();try{if(8&this.f)return;if(void 0===this.x)return;const n=this.x();if("function"==typeof n)this.u=n}finally{t()}};k.prototype.S=function(){if(1&this.f)t();this.f|=1;this.f&=-9;m(this);p(this);u++;const n=i;i=this;return b.bind(this,n)};k.prototype.N=function(){if(!(2&this.f)){this.f|=2;this.o=_;_=this}};k.prototype.d=function(){this.f|=8;if(!(1&this.f))g(this)};function S(t){const n=new k(t);try{n.c()}catch(t){n.d();throw t}return n.d.bind(n)}var x,w,C,E,U,H,N,P,$,D={},T=[],V=/acit|ex(?:s|g|n|p|$)|rph|grid|ows|mnc|ntw|ine[ch]|zoo|^ord|itera/i,A=Array.isArray;function F(t,n){for(var e in n)t[e]=n[e];return t}function M(t){var n=t.parentNode;n&&n.removeChild(t)}function W(t,n,e){var i,_,o,r={};for(o in n)"key"==o?i=n[o]:"ref"==o?_=n[o]:r[o]=n[o];if(arguments.length>2&&(r.children=arguments.length>3?x.call(arguments,2):e),"function"==typeof t&&null!=t.defaultProps)for(o in t.defaultProps)void 0===r[o]&&(r[o]=t.defaultProps[o]);return O(t,r,i,_,null)}function O(t,n,e,i,_){var o={type:t,props:n,key:e,ref:i,__k:null,__:null,__b:0,__e:null,__d:void 0,__c:null,__h:null,constructor:void 0,__v:null==_?++C:_};return null==_&&null!=w.vnode&&w.vnode(o),o}function L(){return{current:null}}function R(t){return t.children}function I(t,n){this.props=t,this.context=n}function j(t,n){if(null==n)return t.__?j(t.__,t.__.__k.indexOf(t)+1):null;for(var e;n<t.__k.length;n++)if(null!=(e=t.__k[n])&&null!=e.__e)return e.__e;return"function"==typeof t.type?j(t):null}function B(t){var n,e;if(null!=(t=t.__)&&null!=t.__c){for(t.__e=t.__c.base=null,n=0;n<t.__k.length;n++)if(null!=(e=t.__k[n])&&null!=e.__e){t.__e=t.__c.base=e.__e;break}return B(t)}}function q(t){(!t.__d&&(t.__d=!0)&&U.push(t)&&!G.__r++||H!==w.debounceRendering)&&((H=w.debounceRendering)||N)(G)}function G(){var t,n,e,i,_,o,r,u,f;for(U.sort(P);t=U.shift();)t.__d&&(n=U.length,i=void 0,_=void 0,o=void 0,u=(r=(e=t).__v).__e,(f=e.__P)&&(i=[],_=[],(o=F({},r)).__v=r.__v+1,it(f,r,o,e.__n,void 0!==f.ownerSVGElement,null!=r.__h?[u]:null,i,null==u?j(r):u,r.__h,_),_t(i,r,_),r.__e!=u&&B(r)),U.length>n&&U.sort(P));G.__r=0}function z(t,n,e,i,_,o,r,u,f,l,s){var c,h,a,p,d,v,y,m,g,b,k=0,S=i&&i.__k||T,x=S.length,w=x,C=n.length;for(e.__k=[],c=0;c<C;c++)null!=(p=e.__k[c]=null==(p=n[c])||"boolean"==typeof p||"function"==typeof p?null:"string"==typeof p||"number"==typeof p||"bigint"==typeof p?O(null,p,null,null,p):A(p)?O(R,{children:p},null,null,null):p.__b>0?O(p.type,p.props,p.key,p.ref?p.ref:null,p.__v):p)&&(p.__=e,p.__b=e.__b+1,-1===(m=X(p,S,y=c+k,w))?a=D:(a=S[m]||D,S[m]=void 0,w--),it(t,p,a,_,o,r,u,f,l,s),d=p.__e,(h=p.ref)&&a.ref!=h&&(a.ref&&rt(a.ref,null,p),s.push(h,p.__c||d,p)),null!=d&&(null==v&&(v=d),b=!(g=a===D||null===a.__v)&&m===y,g?-1==m&&k--:m!==y&&(m===y+1?(k++,b=!0):m>y?w>C-y?(k+=m-y,b=!0):k--:k=m<y&&m==y-1?m-y:0),y=c+k,b=b||m==c&&!g,"function"!=typeof p.type||m===y&&a.__k!==p.__k?"function"==typeof p.type||b?void 0!==p.__d?(f=p.__d,p.__d=void 0):f=d.nextSibling:f=Q(t,d,f):f=J(p,f,t),"function"==typeof e.type&&(e.__d=f)));for(e.__e=v,c=x;c--;)null!=S[c]&&("function"==typeof e.type&&null!=S[c].__e&&S[c].__e==e.__d&&(e.__d=S[c].__e.nextSibling),ut(S[c],S[c]))}function J(t,n,e){for(var i,_=t.__k,o=0;_&&o<_.length;o++)(i=_[o])&&(i.__=t,n="function"==typeof i.type?J(i,n,e):Q(e,i.__e,n));return n}function K(t,n){return n=n||[],null==t||"boolean"==typeof t||(A(t)?t.some((function(t){K(t,n)})):n.push(t)),n}function Q(t,n,e){return null==e||e.parentNode!==t?t.insertBefore(n,null):n==e&&null!=n.parentNode||t.insertBefore(n,e),n.nextSibling}function X(t,n,e,i){var _=t.key,o=t.type,r=e-1,u=e+1,f=n[e];if(null===f||f&&_==f.key&&o===f.type)return e;if(i>(null!=f?1:0))for(;r>=0||u<n.length;){if(r>=0){if((f=n[r])&&_==f.key&&o===f.type)return r;r--}if(u<n.length){if((f=n[u])&&_==f.key&&o===f.type)return u;u++}}return-1}function Y(t,n,e,i,_){var o;for(o in e)"children"===o||"key"===o||o in n||tt(t,o,null,e[o],i);for(o in n)_&&"function"!=typeof n[o]||"children"===o||"key"===o||"value"===o||"checked"===o||e[o]===n[o]||tt(t,o,n[o],e[o],i)}function Z(t,n,e){"-"===n[0]?t.setProperty(n,null==e?"":e):t[n]=null==e?"":"number"!=typeof e||V.test(n)?e:e+"px"}function tt(t,n,e,i,_){var o;t:if("style"===n)if("string"==typeof e)t.style.cssText=e;else{if("string"==typeof i&&(t.style.cssText=i=""),i)for(n in i)e&&n in e||Z(t.style,n,"");if(e)for(n in e)i&&e[n]===i[n]||Z(t.style,n,e[n])}else if("o"===n[0]&&"n"===n[1])o=n!==(n=n.replace(/Capture$/,"")),n=n.toLowerCase()in t?n.toLowerCase().slice(2):n.slice(2),t.l||(t.l={}),t.l[n+o]=e,e?i||t.addEventListener(n,o?et:nt,o):t.removeEventListener(n,o?et:nt,o);else if("dangerouslySetInnerHTML"!==n){if(_)n=n.replace(/xlink(H|:h)/,"h").replace(/sName$/,"s");else if("width"!==n&&"height"!==n&&"href"!==n&&"list"!==n&&"form"!==n&&"tabIndex"!==n&&"download"!==n&&"rowSpan"!==n&&"colSpan"!==n&&n in t)try{t[n]=null==e?"":e;break t}catch(t){}"function"==typeof e||(null==e||!1===e&&"-"!==n[4]?t.removeAttribute(n):t.setAttribute(n,e))}}function nt(t){return this.l[t.type+!1](w.event?w.event(t):t)}function et(t){return this.l[t.type+!0](w.event?w.event(t):t)}function it(t,n,e,i,_,o,r,u,f,l){var s,c,h,a,p,d,v,y,m,g,b,k,S,x,C,E=n.type;if(void 0!==n.constructor)return null;null!=e.__h&&(f=e.__h,u=n.__e=e.__e,n.__h=null,o=[u]),(s=w.__b)&&s(n);try{t:if("function"==typeof E){if(y=n.props,m=(s=E.contextType)&&i[s.__c],g=s?m?m.props.value:s.__:i,e.__c?v=(c=n.__c=e.__c).__=c.__E:("prototype"in E&&E.prototype.render?n.__c=c=new E(y,g):(n.__c=c=new I(y,g),c.constructor=E,c.render=ft),m&&m.sub(c),c.props=y,c.state||(c.state={}),c.context=g,c.__n=i,h=c.__d=!0,c.__h=[],c._sb=[]),null==c.__s&&(c.__s=c.state),null!=E.getDerivedStateFromProps&&(c.__s==c.state&&(c.__s=F({},c.__s)),F(c.__s,E.getDerivedStateFromProps(y,c.__s))),a=c.props,p=c.state,c.__v=n,h)null==E.getDerivedStateFromProps&&null!=c.componentWillMount&&c.componentWillMount(),null!=c.componentDidMount&&c.__h.push(c.componentDidMount);else{if(null==E.getDerivedStateFromProps&&y!==a&&null!=c.componentWillReceiveProps&&c.componentWillReceiveProps(y,g),!c.__e&&(null!=c.shouldComponentUpdate&&!1===c.shouldComponentUpdate(y,c.__s,g)||n.__v===e.__v)){for(n.__v!==e.__v&&(c.props=y,c.state=c.__s,c.__d=!1),n.__e=e.__e,n.__k=e.__k,n.__k.forEach((function(t){t&&(t.__=n)})),b=0;b<c._sb.length;b++)c.__h.push(c._sb[b]);c._sb=[],c.__h.length&&r.push(c);break t}null!=c.componentWillUpdate&&c.componentWillUpdate(y,c.__s,g),null!=c.componentDidUpdate&&c.__h.push((function(){c.componentDidUpdate(a,p,d)}))}if(c.context=g,c.props=y,c.__P=t,c.__e=!1,k=w.__r,S=0,"prototype"in E&&E.prototype.render){for(c.state=c.__s,c.__d=!1,k&&k(n),s=c.render(c.props,c.state,c.context),x=0;x<c._sb.length;x++)c.__h.push(c._sb[x]);c._sb=[]}else do{c.__d=!1,k&&k(n),s=c.render(c.props,c.state,c.context),c.state=c.__s}while(c.__d&&++S<25);c.state=c.__s,null!=c.getChildContext&&(i=F(F({},i),c.getChildContext())),h||null==c.getSnapshotBeforeUpdate||(d=c.getSnapshotBeforeUpdate(a,p)),z(t,A(C=null!=s&&s.type===R&&null==s.key?s.props.children:s)?C:[C],n,e,i,_,o,r,u,f,l),c.base=n.__e,n.__h=null,c.__h.length&&r.push(c),v&&(c.__E=c.__=null)}else null==o&&n.__v===e.__v?(n.__k=e.__k,n.__e=e.__e):n.__e=ot(e.__e,n,e,i,_,o,r,f,l);(s=w.diffed)&&s(n)}catch(t){n.__v=null,(f||null!=o)&&(n.__e=u,n.__h=!!f,o[o.indexOf(u)]=null),w.__e(t,n,e)}}function _t(t,n,e){for(var i=0;i<e.length;i++)rt(e[i],e[++i],e[++i]);w.__c&&w.__c(n,t),t.some((function(n){try{t=n.__h,n.__h=[],t.some((function(t){t.call(n)}))}catch(t){w.__e(t,n.__v)}}))}function ot(t,n,e,i,_,o,r,u,f){var l,s,c,h=e.props,a=n.props,p=n.type,d=0;if("svg"===p&&(_=!0),null!=o)for(;d<o.length;d++)if((l=o[d])&&"setAttribute"in l==!!p&&(p?l.localName===p:3===l.nodeType)){t=l,o[d]=null;break}if(null==t){if(null===p)return document.createTextNode(a);t=_?document.createElementNS("http://www.w3.org/2000/svg",p):document.createElement(p,a.is&&a),o=null,u=!1}if(null===p)h===a||u&&t.data===a||(t.data=a);else{if(o=o&&x.call(t.childNodes),s=(h=e.props||D).dangerouslySetInnerHTML,c=a.dangerouslySetInnerHTML,!u){if(null!=o)for(h={},d=0;d<t.attributes.length;d++)h[t.attributes[d].name]=t.attributes[d].value;(c||s)&&(c&&(s&&c.__html==s.__html||c.__html===t.innerHTML)||(t.innerHTML=c&&c.__html||""))}if(Y(t,a,h,_,u),c)n.__k=[];else if(z(t,A(d=n.props.children)?d:[d],n,e,i,_&&"foreignObject"!==p,o,r,o?o[0]:e.__k&&j(e,0),u,f),null!=o)for(d=o.length;d--;)null!=o[d]&&M(o[d]);u||("value"in a&&void 0!==(d=a.value)&&(d!==t.value||"progress"===p&&!d||"option"===p&&d!==h.value)&&tt(t,"value",d,h.value,!1),"checked"in a&&void 0!==(d=a.checked)&&d!==t.checked&&tt(t,"checked",d,h.checked,!1))}return t}function rt(t,n,e){try{"function"==typeof t?t(n):t.current=n}catch(t){w.__e(t,e)}}function ut(t,n,e){var i,_;if(w.unmount&&w.unmount(t),(i=t.ref)&&(i.current&&i.current!==t.__e||rt(i,null,n)),null!=(i=t.__c)){if(i.componentWillUnmount)try{i.componentWillUnmount()}catch(t){w.__e(t,n)}i.base=i.__P=null,t.__c=void 0}if(i=t.__k)for(_=0;_<i.length;_++)i[_]&&ut(i[_],n,e||"function"!=typeof t.type);e||null==t.__e||M(t.__e),t.__=t.__e=t.__d=void 0}function ft(t,n,e){return this.constructor(t,e)}function lt(t,n,e){var i,_,o,r;w.__&&w.__(t,n),_=(i="function"==typeof e)?null:e&&e.__k||n.__k,o=[],r=[],it(n,t=(!i&&e||n).__k=W(R,null,[t]),_||D,D,void 0!==n.ownerSVGElement,!i&&e?[e]:_?null:n.firstChild?x.call(n.childNodes):null,o,!i&&e?e:_?_.__e:n.firstChild,i,r),_t(o,t,r)}function st(t,n){lt(t,n,st)}function ct(t,n,e){var i,_,o,r,u=F({},t.props);for(o in t.type&&t.type.defaultProps&&(r=t.type.defaultProps),n)"key"==o?i=n[o]:"ref"==o?_=n[o]:u[o]=void 0===n[o]&&void 0!==r?r[o]:n[o];return arguments.length>2&&(u.children=arguments.length>3?x.call(arguments,2):e),O(t.type,u,i||t.key,_||t.ref,null)}function ht(t,n){var e={__c:n="__cC"+$++,__:t,Consumer:function(t,n){return t.children(n)},Provider:function(t){var e,i;return this.getChildContext||(e=[],(i={})[n]=this,this.getChildContext=function(){return i},this.shouldComponentUpdate=function(t){this.props.value!==t.value&&e.some((function(t){t.__e=!0,q(t)}))},this.sub=function(t){e.push(t);var n=t.componentWillUnmount;t.componentWillUnmount=function(){e.splice(e.indexOf(t),1),n&&n.call(t)}}),t.children}};return e.Provider.__=e.Consumer.contextType=e}x=T.slice,w={__e:function(t,n,e,i){for(var _,o,r;n=n.__;)if((_=n.__c)&&!_.__)try{if((o=_.constructor)&&null!=o.getDerivedStateFromError&&(_.setState(o.getDerivedStateFromError(t)),r=_.__d),null!=_.componentDidCatch&&(_.componentDidCatch(t,i||{}),r=_.__d),r)return _.__E=_}catch(n){t=n}throw t}},C=0,E=function(t){return null!=t&&void 0===t.constructor},I.prototype.setState=function(t,n){var e;e=null!=this.__s&&this.__s!==this.state?this.__s:this.__s=F({},this.state),"function"==typeof t&&(t=t(F({},e),this.props)),t&&F(e,t),null!=t&&this.__v&&(n&&this._sb.push(n),q(this))},I.prototype.forceUpdate=function(t){this.__v&&(this.__e=!0,t&&this.__h.push(t),q(this))},I.prototype.render=R,U=[],N="function"==typeof Promise?Promise.prototype.then.bind(Promise.resolve()):setTimeout,P=function(t,n){return t.__v.__b-n.__v.__b},G.__r=0,$=0;var at,pt,dt,vt,yt=0,mt=[],gt=[],bt=w.__b,kt=w.__r,St=w.diffed,xt=w.__c,wt=w.unmount;function Ct(t,n){w.__h&&w.__h(pt,t,yt||n),yt=0;var e=pt.__H||(pt.__H={__:[],__h:[]});return t>=e.__.length&&e.__.push({__V:gt}),e.__[t]}function Et(t){return yt=1,Ut(Bt,t)}function Ut(t,n,e){var i=Ct(at++,2);if(i.t=t,!i.__c&&(i.__=[e?e(n):Bt(void 0,n),function(t){var n=i.__N?i.__N[0]:i.__[0],e=i.t(n,t);n!==e&&(i.__N=[e,i.__[1]],i.__c.setState({}))}],i.__c=pt,!pt.u)){var _=function(t,n,e){if(!i.__c.__H)return!0;var _=i.__c.__H.__.filter((function(t){return t.__c}));if(_.every((function(t){return!t.__N})))return!o||o.call(this,t,n,e);var r=!1;return _.forEach((function(t){if(t.__N){var n=t.__[0];t.__=t.__N,t.__N=void 0,n!==t.__[0]&&(r=!0)}})),!(!r&&i.__c.props===t)&&(!o||o.call(this,t,n,e))};pt.u=!0;var o=pt.shouldComponentUpdate,r=pt.componentWillUpdate;pt.componentWillUpdate=function(t,n,e){if(this.__e){var i=o;o=void 0,_(t,n,e),o=i}r&&r.call(this,t,n,e)},pt.shouldComponentUpdate=_}return i.__N||i.__}function Ht(t,n){var e=Ct(at++,3);!w.__s&&jt(e.__H,n)&&(e.__=t,e.i=n,pt.__H.__h.push(e))}function Nt(t,n){var e=Ct(at++,4);!w.__s&&jt(e.__H,n)&&(e.__=t,e.i=n,pt.__h.push(e))}function Pt(t){return yt=5,Dt((function(){return{current:t}}),[])}function $t(t,n,e){yt=6,Nt((function(){return"function"==typeof t?(t(n()),function(){return t(null)}):t?(t.current=n(),function(){return t.current=null}):void 0}),null==e?e:e.concat(t))}function Dt(t,n){var e=Ct(at++,7);return jt(e.__H,n)?(e.__V=t(),e.i=n,e.__h=t,e.__V):e.__}function Tt(t,n){return yt=8,Dt((function(){return t}),n)}function Vt(t){var n=pt.context[t.__c],e=Ct(at++,9);return e.c=t,n?(null==e.__&&(e.__=!0,n.sub(pt)),n.props.value):t.__}function At(t,n){w.useDebugValue&&w.useDebugValue(n?n(t):t)}function Ft(t){var n=Ct(at++,10),e=Et();return n.__=t,pt.componentDidCatch||(pt.componentDidCatch=function(t,i){n.__&&n.__(t,i),e[1](t)}),[e[0],function(){e[1](void 0)}]}function Mt(){var t=Ct(at++,11);if(!t.__){for(var n=pt.__v;null!==n&&!n.__m&&null!==n.__;)n=n.__;var e=n.__m||(n.__m=[0,0]);t.__="P"+e[0]+"-"+e[1]++}return t.__}function Wt(){for(var t;t=mt.shift();)if(t.__P&&t.__H)try{t.__H.__h.forEach(Rt),t.__H.__h.forEach(It),t.__H.__h=[]}catch(u){t.__H.__h=[],w.__e(u,t.__v)}}w.__b=function(t){pt=null,bt&&bt(t)},w.__r=function(t){kt&&kt(t),at=0;var n=(pt=t.__c).__H;n&&(dt===pt?(n.__h=[],pt.__h=[],n.__.forEach((function(t){t.__N&&(t.__=t.__N),t.__V=gt,t.__N=t.i=void 0}))):(n.__h.forEach(Rt),n.__h.forEach(It),n.__h=[],at=0)),dt=pt},w.diffed=function(t){St&&St(t);var n=t.__c;n&&n.__H&&(n.__H.__h.length&&(1!==mt.push(n)&&vt===w.requestAnimationFrame||((vt=w.requestAnimationFrame)||Lt)(Wt)),n.__H.__.forEach((function(t){t.i&&(t.__H=t.i),t.__V!==gt&&(t.__=t.__V),t.i=void 0,t.__V=gt}))),dt=pt=null},w.__c=function(t,n){n.some((function(t){try{t.__h.forEach(Rt),t.__h=t.__h.filter((function(t){return!t.__||It(t)}))}catch(s){n.some((function(t){t.__h&&(t.__h=[])})),n=[],w.__e(s,t.__v)}})),xt&&xt(t,n)},w.unmount=function(t){wt&&wt(t);var n,e=t.__c;e&&e.__H&&(e.__H.__.forEach((function(t){try{Rt(t)}catch(t){n=t}})),e.__H=void 0,n&&w.__e(n,e.__v))};var Ot="function"==typeof requestAnimationFrame;function Lt(t){var n,e=function(){clearTimeout(i),Ot&&cancelAnimationFrame(n),setTimeout(t)},i=setTimeout(e,100);Ot&&(n=requestAnimationFrame(e))}function Rt(t){var n=pt,e=t.__c;"function"==typeof e&&(t.__c=void 0,e()),pt=n}function It(t){var n=pt;t.__c=t.__(),pt=n}function jt(t,n){return!t||t.length!==n.length||n.some((function(n,e){return n!==t[e]}))}function Bt(t,n){return"function"==typeof n?n(t):n}function qt(t,n){w[t]=n.bind(null,w[t]||(()=>{}))}let Gt,zt;function Jt(t){if(zt)zt();zt=t&&t.S()}function Kt({data:t}){const n=Xt(t);n.value=t;const e=Dt(()=>{let t=this.__v;while(t=t.__)if(t.__c){t.__c.__$f|=4;break}this.__$u.c=()=>{var t;if(!E(e.peek())&&3===(null==(t=this.base)?void 0:t.nodeType))this.base.data=e.peek();else{this.__$f|=1;this.setState({})}};return y(()=>{let t=n.value.value;return 0===t?0:!0===t?"":t||""})},[]);return e.value}Kt.displayName="_st";Object.defineProperties(c.prototype,{constructor:{configurable:!0,value:void 0},type:{configurable:!0,value:Kt},props:{configurable:!0,get(){return{data:this}}},__b:{configurable:!0,value:1}});qt("__b",(t,n)=>{if("string"==typeof n.type){let t,e=n.props;for(let i in e){if("children"===i)continue;let _=e[i];if(_ instanceof c){if(!t)n.__np=t={};t[i]=_;e[i]=_.peek()}}}t(n)});qt("__r",(t,n)=>{Jt();let e,i=n.__c;if(i){i.__$f&=-2;e=i.__$u;if(void 0===e)i.__$u=e=function(t){let n;S((function(){n=this}));n.c=()=>{i.__$f|=1;i.setState({})};return n}()}Gt=i;Jt(e);t(n)});qt("__e",(t,n,e,i)=>{Jt();Gt=void 0;t(n,e,i)});qt("diffed",(t,n)=>{Jt();Gt=void 0;let e;if("string"==typeof n.type&&(e=n.__e)){let t=n.__np,i=n.props;if(t){let n=e.U;if(n)for(let e in n){let i=n[e];if(void 0!==i&&!(e in t)){i.d();n[e]=void 0}}else{n={};e.U=n}for(let _ in t){let o=n[_],r=t[_];if(void 0===o){o=Qt(e,_,r,i);n[_]=o}else o.o(r,i)}}}t(n)});function Qt(t,n,e,i){const _=n in t&&void 0===t.ownerSVGElement,o=h(e);return{o:(t,n)=>{o.value=t;i=n},d:S(()=>{const e=o.value.value;if(i[n]!==e){i[n]=e;if(_)t[n]=e;else if(e)t.setAttribute(n,e);else t.removeAttribute(n)}})}}qt("unmount",(t,n)=>{if("string"==typeof n.type){let t=n.__e;if(t){const n=t.U;if(n){t.U=void 0;for(let t in n){let e=n[t];if(e)e.d()}}}}else{let t=n.__c;if(t){const n=t.__$u;if(n){t.__$u=void 0;n.d()}}}t(n)});qt("__h",(t,n,e,i)=>{if(i<3||9===i)n.__$f|=2;t(n,e,i)});I.prototype.shouldComponentUpdate=function(t,n){const e=this.__$u;if(!(e&&void 0!==e.s||4&this.__$f))return!0;if(3&this.__$f)return!0;for(let i in n)return!0;for(let i in t)if("__source"!==i&&t[i]!==this.props[i])return!0;for(let i in this.props)if(!(i in t))return!0;return!1};function Xt(t){return Dt(()=>h(t),[])}function Yt(t){const n=Pt(t);n.current=t;Gt.__$f|=4;return Dt(()=>y(()=>n.current()),[])}function Zt(t){const n=Pt(t);n.current=t;Ht(()=>S(()=>n.current()),[])}var tn=function(t,n,e,i){var _;n[0]=0;for(var o=1;o<n.length;o++){var r=n[o++],u=n[o]?(n[0]|=r?1:2,e[n[o++]]):n[++o];3===r?i[0]=u:4===r?i[1]=Object.assign(i[1]||{},u):5===r?(i[1]=i[1]||{})[n[++o]]=u:6===r?i[1][n[++o]]+=u+"":r?(_=t.apply(u,tn(t,u,e,["",null])),i.push(_),u[0]?n[0]|=2:(n[o-2]=0,n[o]=_)):i.push(u)}return i},nn=new Map;function en(t){var n=nn.get(this);return n||(n=new Map,nn.set(this,n)),(n=tn(this,n.get(t)||(n.set(t,n=function(t){for(var n,e,i=1,_="",o="",r=[0],u=function(t){1===i&&(t||(_=_.replace(/^\s*\n\s*|\s*\n\s*$/g,"")))?r.push(0,t,_):3===i&&(t||_)?(r.push(3,t,_),i=2):2===i&&"..."===_&&t?r.push(4,t,0):2===i&&_&&!t?r.push(5,0,!0,_):i>=5&&((_||!t&&5===i)&&(r.push(i,0,_,e),i=6),t&&(r.push(i,t,0,e),i=6)),_=""},f=0;f<t.length;f++){f&&(1===i&&u(),u(f));for(var l=0;l<t[f].length;l++)n=t[f][l],1===i?"<"===n?(u(),r=[r],i=3):_+=n:4===i?"--"===_&&">"===n?(i=1,_=""):_=n+_[0]:o?n===o?o="":_+=n:'"'===n||"'"===n?o=n:">"===n?(u(),i=1):i&&("="===n?(i=5,e=_,_=""):"/"===n&&(i<5||">"===t[f][l+1])?(u(),3===i&&(r=r[0]),i=r,(r=r[0]).push(2,0,i),i=0):" "===n||"\t"===n||"\n"===n||"\r"===n?(u(),i=2):_+=n),3===i&&"!--"===_&&(i=4,r=r[0])}return u(),r}(t)),n),arguments,[])).length>1?n:n[0]}var _n=en.bind(W);export{I as Component,R as Fragment,c as Signal,e as batch,ct as cloneElement,y as computed,ht as createContext,W as createElement,L as createRef,S as effect,W as h,_n as html,st as hydrate,E as isValidElement,w as options,lt as render,h as signal,K as toChildArray,r as untracked,Tt as useCallback,Yt as useComputed,Vt as useContext,At as useDebugValue,Ht as useEffect,Ft as useErrorBoundary,Mt as useId,$t as useImperativeHandle,Nt as useLayoutEffect,Dt as useMemo,Ut as useReducer,Pt as useRef,Xt as useSignal,Zt as useSignalEffect,Et as useState};
